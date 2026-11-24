// Blender NURBS add-on extension to the USD BRep Python bindings.
//
// This file replaces SmPyUsd.cpp in our macOS build (no OpenUSD there). It
// defines bind_usd() so the upstream SmPyMain.cpp links unchanged, and uses
// it to register a `blender_ext` submodule with the few kernel entry points
// the upstream Python API does not expose:
//   - build a B-spline surface / curve from explicit CVs, knots and weights
//   - read CVs, knots and weights back from curves and surfaces
//   - wrap a surface into a single-face Brep (for tessellation / export)
//   - identify fillet faces and remove picked face groups (Remove Fillets)

#include "SmPyCommon.h"

#include <SmApiGeneral.h>
#include <SmApiCurves.h>
#include <SmBSplineCurve.h>
#include <SmBSplineSurface.h>
#include <SmBrep.h>
#include <SmBrepData.h>
#include <SmContext.h>
#include <SmFace.h>
#include <SmFaceuse.h>
#include <SmEdgeuse.h>
#include <SmEdge.h>
#include <SmTrimmingTools.h>
#include <SmVertex.h>
#include <SmFeatureExecutive.h>
#include <SmExtent1d.h>
#include <algorithm>

#include <cmath>
#include <chrono>
#include <memory>
#include <cstdint>
#include <vector>

// Sparse linear algebra for the add-on's solvers (SubD to BREP): Eigen's
// sparse LDL^T with its fill-reducing ordering. Only Eigen's MPL2 code is used.
#define EIGEN_MPL2_ONLY
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include <cstdlib>
#include <pybind11/numpy.h>

namespace
{

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int64_t>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

// C (m x n) from triplets (duplicates add up).
SpMat sparse_from(const I64& rows, const I64& cols, const F64& vals, std::int64_t m, std::int64_t n)
{
    const auto nnz = vals.size();
    if (rows.size() != nnz || cols.size() != nnz)
        throw std::invalid_argument("rows, cols and vals differ in length");
    const auto* r = rows.data();
    const auto* c = cols.data();
    const auto* v = vals.data();
    std::vector<Eigen::Triplet<double, std::int64_t>> t;
    t.reserve(static_cast<size_t>(nnz));
    for (py::ssize_t k = 0; k < nnz; ++k)
    {
        if (r[k] < 0 || r[k] >= m || c[k] < 0 || c[k] >= n)
            throw std::out_of_range("triplet index out of range");
        t.emplace_back(r[k], c[k], v[k]);
    }
    SpMat C(m, n);
    C.setFromTriplets(t.begin(), t.end());
    return C;
}

py::array_t<double> to_numpy(const Eigen::VectorXd& x)
{
    py::array_t<double> out(x.size());
    std::copy(x.data(), x.data() + x.size(), out.mutable_data());
    return out;
}

// Split a full knot vector into unique values + multiplicities.
void split_knots(const std::vector<double>& full, SmTArray<double>& uniq, SmTArray<ULONG>& mult)
{
    const double tol = 1e-12;
    for (double k : full)
    {
        if (uniq.GetSize() > 0 && std::fabs(uniq[uniq.GetSize() - 1] - k) <= tol)
            mult[mult.GetSize() - 1] += 1;
        else
        {
            uniq.Add(k);
            mult.Add(1);
        }
    }
}

py::list expand_knots(const SmTArray<double>& uniq, const SmTArray<ULONG>& mult)
{
    py::list out;
    for (ULONG i = 0; i < uniq.GetSize(); ++i)
        for (ULONG j = 0; j < mult[i]; ++j)
            out.append(uniq[i]);
    return out;
}

SmBSplineSurface* as_bspline_surface(SmSurface* s)
{
    auto* b = dynamic_cast<SmBSplineSurface*>(s);
    if (!b)
        throw std::invalid_argument("surface is not a B-spline surface");
    return b;
}

SmBSplineCurve* as_bspline_curve(SmCurve* c)
{
    auto* b = dynamic_cast<SmBSplineCurve*>(c);
    if (!b)
        throw std::invalid_argument("curve is not a B-spline curve");
    return b;
}

// The quasi-definite system K = [[M, C^T], [C, -delta s I]] (s: M's mean diagonal), factorised once and
// solved many times; each solve refines against delta = 0. Supernodal LDL^T from Accelerate on macOS,
// Eigen's simplicial LDL^T elsewhere (or on request).
class KKTFactor
{
public:
    KKTFactor(const SpMat& M, const SpMat& C, double delta, bool eigen_only)
        : n_(M.rows()), m_(C.rows())
    {
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        std::vector<Eigen::Triplet<double, std::int64_t>> t;
        t.reserve(static_cast<size_t>(M.nonZeros() + 2 * C.nonZeros() + m_));
        double scale = 0.0;
        for (std::int64_t k = 0; k < M.outerSize(); ++k)
            for (SpMat::InnerIterator it(M, k); it; ++it)
            {
                t.emplace_back(it.row(), it.col(), it.value());
                if (it.row() == it.col())
                    scale += it.value();
            }
        scale = n_ > 0 ? scale / static_cast<double>(n_) : 1.0;
        for (std::int64_t k = 0; k < C.outerSize(); ++k)
            for (SpMat::InnerIterator it(C, k); it; ++it)
            {
                t.emplace_back(n_ + it.row(), it.col(), it.value());
                t.emplace_back(it.col(), n_ + it.row(), it.value());
            }
        reg_ = delta * scale;
        for (std::int64_t i = 0; i < m_; ++i)
            t.emplace_back(n_ + i, n_ + i, -reg_);
        K_ = SpMat(n_ + m_, n_ + m_);
        K_.setFromTriplets(t.begin(), t.end());
        const SpMat& K = K_;
        auto t1 = clock::now();
#ifdef __APPLE__
        if (!eigen_only)
        {
            const std::int64_t N = n_ + m_;
            cs_.assign(N + 1, 0);
            ri_.reserve(static_cast<size_t>(K.nonZeros() / 2 + N));
            va_.reserve(ri_.capacity());
            for (std::int64_t k = 0; k < N; ++k)
            {
                for (SpMat::InnerIterator it(K, k); it; ++it)
                    if (it.row() >= k)
                    {
                        ri_.push_back(static_cast<int>(it.row()));
                        va_.push_back(it.value());
                    }
                cs_[k + 1] = static_cast<long>(ri_.size());
            }
            SparseMatrixStructure st{};
            st.rowCount = static_cast<int>(N);
            st.columnCount = static_cast<int>(N);
            st.columnStarts = cs_.data();
            st.rowIndices = ri_.data();
            st.attributes.kind = SparseSymmetric;
            st.attributes.triangle = SparseLowerTriangle;
            st.blockSize = 1;
            SparseSymbolicFactorOptions so{};
            so.control = SparseDefaultControl;
            so.orderMethod = SparseOrderDefault;
            so.malloc = [](size_t sz) { return std::malloc(sz); };
            so.free = std::free;
            sym_ = SparseFactor(SparseFactorizationLDLTUnpivoted, st, so);
            if (sym_.status != SparseStatusOK)
                throw std::runtime_error("KKT analysis failed");
            auto t2 = clock::now();
            SparseMatrix_Double A{st, va_.data()};
            fac_ = SparseFactor(sym_, A);
            if (fac_.status != SparseStatusOK)
            {
                SparseCleanup(sym_);
                throw std::runtime_error("KKT factorisation failed");
            }
            accelerate_ = true;
            auto t3 = clock::now();
            times_ = {sec(t1 - t0), sec(t2 - t1), sec(t3 - t2)};
            return;
        }
#endif
        ldlt_ = std::make_unique<Eigen::SimplicialLDLT<SpMat>>();
        ldlt_->analyzePattern(K);
        auto t2 = clock::now();
        ldlt_->factorize(K);
        if (ldlt_->info() != Eigen::Success)
            throw std::runtime_error("KKT factorisation failed");
        auto t3 = clock::now();
        times_ = {sec(t1 - t0), sec(t2 - t1), sec(t3 - t2)};
    }
    ~KKTFactor()
    {
#ifdef __APPLE__
        if (accelerate_)
        {
            SparseCleanup(fac_);
            SparseCleanup(sym_);
        }
#endif
    }
    KKTFactor(const KKTFactor&) = delete;
    KKTFactor& operator=(const KKTFactor&) = delete;

    // [x; y] for the right side [rx; ry], with a few refinement steps against the delta = 0 system
    Eigen::VectorXd solve(const Eigen::VectorXd& b, int refine = 3) const
    {
        Eigen::VectorXd s = raw(b);
        for (int it = 0; it < refine; ++it)
        {
            Eigen::VectorXd r = b - K_ * s;                    // against delta = 0: K0 s = K s + reg [0; s_y]
            r.tail(m_) -= reg_ * s.tail(m_);
            s += raw(r);
        }
        return s;
    }
    std::int64_t n() const { return n_; }
    std::int64_t m() const { return m_; }
    bool accelerate() const { return accelerate_; }
    const std::vector<double>& times() const { return times_; }

private:
    static double sec(std::chrono::steady_clock::duration d) { return std::chrono::duration<double>(d).count(); }
    Eigen::VectorXd raw(const Eigen::VectorXd& b) const
    {
#ifdef __APPLE__
        if (accelerate_)
        {
            Eigen::VectorXd out = b;
            DenseVector_Double xv{static_cast<int>(out.size()), out.data()};
            SparseSolve(fac_, xv);
            return out;
        }
#endif
        return ldlt_->solve(b);
    }
    std::int64_t n_, m_;
    SpMat K_;
    double reg_ = 0.0;
    std::unique_ptr<Eigen::SimplicialLDLT<SpMat>> ldlt_;
    bool accelerate_ = false;
    std::vector<double> times_;
#ifdef __APPLE__
    std::vector<long> cs_;
    std::vector<int> ri_;
    std::vector<double> va_;
    SparseOpaqueSymbolicFactorization sym_{};
    SparseOpaqueFactorization_Double fac_{};
#endif
};

} // namespace

void bind_usd(py::module_& m)
{
    py::module_ ext = m.def_submodule("blender_ext", "Extra kernel access for the Blender add-on.");

    ext.def("surface_from_cvs",
        [](ULONG degree_u, ULONG degree_v, std::vector<py::tuple>& points,
           std::vector<double>& knots_u, std::vector<double>& knots_v,
           py::object weights) {
            SmTArray<SmPoint3d> pts;
            for (auto& t : points) pts.Add(to_point(t));
            SmTArray<double> ku, kv;
            SmTArray<ULONG> mu, mv;
            split_knots(knots_u, ku, mu);
            split_knots(knots_v, kv, mv);
            SmTArray<double> w;
            const SmTArray<double>* pw = nullptr;
            if (!weights.is_none())
            {
                for (auto x : weights.cast<std::vector<double>>()) w.Add(x);
                pw = &w;
            }
            SmBSplineSurface* r = nullptr;
            CHECK_STATUS(SmBSplineSurface::CreateCanonical(
                *SmApiGetOrCreateContext(), degree_u, degree_v, pts, SM_SF_UNSPECIFIED,
                mu, mv, ku, kv, SM_KT_UNSPECIFIED, pw, nullptr, r));
            return static_cast<SmSurface*>(r);
        },
        py::return_value_policy::reference,
        py::arg("degree_u"), py::arg("degree_v"), py::arg("points"),
        py::arg("knots_u"), py::arg("knots_v"), py::arg("weights") = py::none(),
        "Create a B-spline surface from CVs and full (clamped) knot vectors.");

    ext.def("surface_cvs",
        [](SmSurface* s) {
            SmBSplineSurface* b = as_bspline_surface(s);
            ULONG nu = 0, nv = 0;
            SmTArray<SmPoint3d> pts;
            SmTArray<double> w;
            CHECK_STATUS(b->GetControlPointNet(nu, nv, pts, w));
            SmTArray<double> ku, kv;
            SmTArray<ULONG> mu, mv;
            CHECK_STATUS(b->GetKnots(SM_SP_U, ku, &mu));
            CHECK_STATUS(b->GetKnots(SM_SP_V, kv, &mv));
            py::list p, wl;
            for (ULONG i = 0; i < pts.GetSize(); ++i) p.append(point_to_tuple(pts[i]));
            for (ULONG i = 0; i < w.GetSize(); ++i) wl.append(w[i]);
            py::dict d;
            d["degree_u"] = b->GetDegree(SM_SP_U);
            d["degree_v"] = b->GetDegree(SM_SP_V);
            d["count_u"] = nu;
            d["count_v"] = nv;
            d["points"] = p;
            d["weights"] = wl;
            d["knots_u"] = expand_knots(ku, mu);
            d["knots_v"] = expand_knots(kv, mv);
            return d;
        },
        py::arg("surface"), "Control net, knots and weights of a B-spline surface.");

    ext.def("curve_from_cvs",
        [](ULONG degree, std::vector<py::tuple>& points, std::vector<double>& knots, py::object weights) {
            SmTArray<SmPoint3d> pts;
            for (auto& t : points) pts.Add(to_point(t));
            SmTArray<double> k;
            SmTArray<ULONG> mlt;
            split_knots(knots, k, mlt);
            SmTArray<double> w;
            const SmTArray<double>* pw = nullptr;
            if (!weights.is_none())
            {
                for (auto x : weights.cast<std::vector<double>>()) w.Add(x);
                pw = &w;
            }
            SmBSplineCurve* r = nullptr;
            CHECK_STATUS(SmApiCreateCanonicalCurve(pts, k, mlt, pw, degree, SM_CF_UNSPECIFIED, r));
            return static_cast<SmCurve*>(r);
        },
        py::return_value_policy::reference,
        py::arg("degree"), py::arg("points"), py::arg("knots"), py::arg("weights") = py::none(),
        "Create a B-spline curve from CVs and a full (clamped) knot vector.");

    ext.def("curve_cvs",
        [](SmCurve* c) {
            SmBSplineCurve* b = as_bspline_curve(c);
            SmTArray<SmPoint3d> pts;
            SmTArray<double> w;
            CHECK_STATUS(b->GetControlPolygon(pts, w));
            SmTArray<double> k;
            SmTArray<ULONG> mlt;
            CHECK_STATUS(b->GetKnots(k, &mlt));
            py::list p, wl;
            for (ULONG i = 0; i < pts.GetSize(); ++i) p.append(point_to_tuple(pts[i]));
            for (ULONG i = 0; i < w.GetSize(); ++i) wl.append(w[i]);
            py::dict d;
            d["degree"] = b->GetDegree();
            d["points"] = p;
            d["weights"] = wl;
            d["knots"] = expand_knots(k, mlt);
            return d;
        },
        py::arg("curve"), "Control polygon, knots and weights of a B-spline curve.");

    ext.def("read_brep_exact",
        [](const std::string& filename) {
            // Same as SmApiReadBrepFromFile(ascii) but WITHOUT bAddAnalytics:
            // the stock reader replaces every surface it recognizes (e.g. a flat
            // cubic B-spline) by an analytic one (SmPlane, degree 1), which loses
            // the CV layout a NURBS modeler must keep.
            SmContext* ctx = SmApiGetOrCreateContext();
            SmBrepData* data = nullptr;
            SmStatus st = SmBrepData::ReadFromFile(*ctx, (const TCHAR*)filename.c_str(), data, SM_ASCII);
            if (st != SM_SUCCESS)
            {
                delete data;
                SmPyRaise(st, "SmBrepData::ReadFromFile", __FILE__, __LINE__);
            }
            SmBrep* brep = new (*ctx) SmBrep();
            SmTArray<SmAttribute*> attrs;
            st = brep->MakeTopologyFromData(data, attrs, FALSE, FALSE, TRUE);
            delete data;
            if (st != SM_SUCCESS)
            {
                delete brep;
                SmPyRaise(st, "SmBrep::MakeTopologyFromData", __FILE__, __LINE__);
            }
            return brep;
        },
        py::return_value_policy::reference, py::arg("filename"),
        "Read a native ASCII Brep file keeping every B-spline as written (no analytic conversion).");

    ext.def("brep_from_surface",
        [](SmSurface* s) {
            SmBrep* brep = new (*SmApiGetOrCreateContext()) SmBrep();
            SmFace* face = nullptr;
            SmStatus st = brep->CreateFaceFromSurface(s, s->GetNaturalUVDomain(), face);
            if (st != SM_SUCCESS)
            {
                if (s->GetOwner() == nullptr)
                    delete brep;
                SmPyRaise(st, "SmBrep::CreateFaceFromSurface", __FILE__, __LINE__);
            }
            return brep;
        },
        py::return_value_policy::reference, py::arg("surface"),
        "Wrap an untrimmed surface into a one-face Brep (the Brep takes ownership).");

    ext.def("replace_face_surfaces_keep_uv",
        [](SmBrep* brep, std::vector<SmSurface*>& newSurfaces) {
            // Give every face of the Brep a new surface (one per face, in
            // GetFaces() order) while keeping all trims fixed in UV space: edges
            // are re-lifted from their UV trim curves onto the new surfaces and
            // vertices follow (Alias-style CV editing of trimmed surfaces).
            SmTArray<SmFace*> faces;
            brep->GetFaces(faces);
            if (faces.GetSize() == 0 || faces.GetSize() != newSurfaces.size())
                throw std::invalid_argument("need exactly one new surface per face");
            SmTemporaryChangeValue<SmBoolean> editing(brep->m_bEditingEnabled, TRUE);

            struct Use { SmEdgeuse* eu; SmBSplineCurve* uv; SmSurface* srf; };
            std::vector<Use> uses;
            for (ULONG fi = 0; fi < faces.GetSize(); ++fi)
            {
                SmFaceuse* fu = faces[fi]->GetUpwardFaceuse();
                if (!fu)
                    throw std::runtime_error("face has no faceuse");
                SmTArray<SmEdgeuse*> eus;
                fu->GetEdgeuses(eus);
                for (ULONG i = 0; i < eus.GetSize(); ++i)
                {
                    SmBSplineCurve* uv = eus[i]->GetUVTrimCurve();
                    if (!uv)
                    {
                        // some faces come without UV trim curves (the kernel drops
                        // them when it trims a surface that is not G1, e.g. a STEP
                        // surface with full-multiplicity inner knots): drop the 3D
                        // edge onto the face's (old) surface now, before it changes
                        SmBSplineCurve* made = nullptr;
                        if (eus[i]->GetOrCreateUVTrimCurve(made) != SM_SUCCESS || made == nullptr)
                            throw std::runtime_error("an edge has no UV trim curve and none could be made");
                        uv = eus[i]->GetUVTrimCurve();
                        if (!uv)
                            uv = made;
                    }
                    uses.push_back({eus[i], new (*SmApiGetOrCreateContext()) SmBSplineCurve(*uv), newSurfaces[fi]});
                }
            }
            for (ULONG fi = 0; fi < faces.GetSize(); ++fi)
                faces[fi]->SetSurface(newSurfaces[fi], TRUE);

            std::vector<SmEdge*> done;
            for (const Use& u : uses)
            {
                SmEdge* edge = u.eu->GetEdge();
                if (std::find(done.begin(), done.end(), edge) != done.end())
                    continue;
                done.push_back(edge);
                SmBSplineCurve* uvForLift = new (*SmApiGetOrCreateContext()) SmBSplineCurve(*u.uv);
                SmTArray<SmBSplineCurve*> lifted;
                SmApiStatus st = SmApiLiftUVCurveFromSrf(uvForLift, u.srf, &lifted);
                delete uvForLift;
                if (st != SM_SUCCESS || lifted.GetSize() != 1)
                    throw std::runtime_error("could not lift a trim edge onto the new surface");
                CHECK_STATUS(edge->Replace3DCurve(lifted[0]));
            }
            for (SmEdge* edge : done)
            {
                SmCurve* c = edge->GetCurve();
                if (!c) continue;
                SmVertex* v0 = nullptr;
                SmVertex* v1 = nullptr;
                edge->GetVertices(v0, v1);
                SmExtent1d ivl = edge->GetInterval();
                if (ivl.GetMax() <= ivl.GetMin())
                    ivl = c->GetNaturalInterval();
                SmPoint3d pa, pb;
                CHECK_STATUS(c->EvaluatePoint(ivl.GetMin(), pa));
                CHECK_STATUS(c->EvaluatePoint(ivl.GetMax(), pb));
                if (v0) v0->SetPoint(pa);
                if (v1 && v1 != v0) v1->SetPoint(pb);
            }
            for (const Use& u : uses)
                u.eu->SetUVTrimCurve(u.uv, 0.0, TRUE);
            return brep;
        },
        py::return_value_policy::reference, py::arg("brep"), py::arg("surfaces"),
        "Replace the surface of every face (one per face), keeping trims fixed in UV space.");

    ext.def("count_missing_uv_trims",
        [](SmBrep* brep) {
            // Edge uses of the Brep's faces that have no UV trim curve (the
            // kernel drops them when it trims a surface that is not G1).
            SmTArray<SmFace*> faces;
            brep->GetFaces(faces);
            int n = 0;
            for (ULONG fi = 0; fi < faces.GetSize(); ++fi)
            {
                SmFaceuse* fu = faces[fi]->GetUpwardFaceuse();
                if (!fu) continue;
                SmTArray<SmEdgeuse*> eus;
                fu->GetEdgeuses(eus);
                for (ULONG i = 0; i < eus.GetSize(); ++i)
                    if (!eus[i]->GetUVTrimCurve()) ++n;
            }
            return n;
        },
        py::arg("brep"), "Number of edge uses without a UV trim curve.");

    ext.def("trim_surface_with_3d_curves_tol",
        [](SmSurface* srf, std::vector<unsigned long>& curveLoops, std::vector<SmCurve*>& curves,
           double fitTolerance, double brepTolerance) {
            // As usd_brep.trim_surface_with_3d_curves, but the UV trim curves
            // are fitted to the 3D curves within fitTolerance (upstream fixes
            // 0.01 model units, which lets a UV curve stray far from its edge on
            // a small part; CV editing re-lifts edges from the UV curves). The
            // Brep's own tolerance is set to brepTolerance afterwards, so later
            // operations (stitch, booleans) see the same value as before.
            if (srf == nullptr || fitTolerance <= 0.0)
                throw std::invalid_argument("need a surface and a positive fit tolerance");
            SmTArray<ULONG> loops;
            SmTArray<SmCurve*> carr;
            SmTArray<SmOrientType> orients;
            SmTArray<SmPoint3d> loopPoints;
            for (auto v : curveLoops) loops.Add(v);
            for (auto* c : curves) { carr.Add(c); orients.Add(SM_OT_SAME); }
            SmExtent2d domain = srf->GetNaturalUVDomain();
            SmBrep* r = nullptr;
            CHECK_STATUS(SmTrimmingTools::TrimSurfaceWithModelSpaceCurves(
                *SmApiGetOrCreateContext(), fitTolerance, loops, carr, orients, loopPoints, srf, domain,
                SM_OT_SAME, TRUE, TRUE, TRUE, r));
            if (r != nullptr && brepTolerance > 0.0)
                r->SetTolerance(brepTolerance);
            return r;
        },
        py::return_value_policy::reference, py::arg("surface"), py::arg("curve_loops"), py::arg("curves"),
        py::arg("fit_tolerance"), py::arg("brep_tolerance") = 0.01,
        "Consumes: surface, curves. Trim a surface with 3D curve loops, UV trim curves within fit_tolerance.");

    ext.def("append_brep",
        [](SmBrep* dst, SmBrep* src) {
            // Copy src's faces / edges / vertices into dst as they are: no
            // intersection or coincidence checks (unlike a MERGE boolean, which
            // imprints faces that touch tangentially). Stitch afterwards.
            if (dst == nullptr || src == nullptr || dst == src)
                throw std::invalid_argument("need two different breps");
            SmTemporaryChangeValue<SmBoolean> editing(dst->m_bEditingEnabled, TRUE);
            CHECK_STATUS(dst->MergeBrep(*src, nullptr, nullptr, FALSE));
            return dst;
        },
        py::return_value_policy::reference, py::arg("dst"), py::arg("src"),
        "Copy every face of src into dst without intersecting (src is left unchanged).");

    ext.def("delete_faces",
        [](SmBrep* brep, std::vector<SmFace*>& faces, bool delete_edges, bool region_nesting) {
            for (SmFace* f : faces)
            {
                if (f == nullptr || f->GetBrep() != brep)
                    throw std::invalid_argument("face does not belong to this brep");
            }
            // Boolean results come back with editing disabled; enable it for the
            // duration of this edit (the pattern documented in SmBrep.h).
            SmTemporaryChangeValue<SmBoolean> editing(brep->m_bEditingEnabled, TRUE);
            for (SmFace* f : faces)
                CHECK_STATUS(brep->DeleteFace(f, delete_edges ? TRUE : FALSE, region_nesting ? TRUE : FALSE));
            return brep;
        },
        py::return_value_policy::reference, py::arg("brep"), py::arg("faces"),
        py::arg("delete_edges") = true, py::arg("region_nesting") = true,
        "Delete faces (and edges/vertices left dangling) from a Brep, in place. "
        "Face handles passed here are stale afterwards.");

    // -- Remove Fillets: the kernel's own fillet recognition and defeaturing, face by
    //    face (the upstream remove_fillet only does the whole Brep, all or nothing).
    ext.def("fillet_info",
        [](SmFace* face, double tol, double tangency_deg) {
            if (face == nullptr)
                throw std::invalid_argument("no face");
            SmBoolean is_fillet = FALSE;
            ULONG xsect = 0, solver = 0;
            SmSurfParamType rail_dir = SM_SP_U;
            double radius = 0.0, dist = 0.0;
            SmTArray<SmEdge*> rails_min, rails_max;
            SmExtent1d radii;
            SmStatus st = face->IdentifyFillet(tol, tangency_deg, TRUE, FALSE, is_fillet, xsect, solver,
                                               rail_dir, radius, dist, rails_min, rails_max, &radii);
            py::dict out;
            const bool ok = st == SM_SUCCESS && is_fillet;
            out["is_fillet"] = ok;
            out["cross_section"] = (int)xsect;
            out["solver"] = (int)solver;
            out["rail_dir"] = rail_dir == SM_SP_U ? "U" : "V";
            out["radius"] = radius;
            out["dist"] = dist;
            const bool varying = ok && solver == 2 && radii.GetMin() <= radii.GetMax();
            out["radius_min"] = varying ? radii.GetMin() : radius;
            out["radius_max"] = varying ? radii.GetMax() : radius;
            std::vector<SmEdge*> a, b;
            for (ULONG i = 0; ok && i < rails_min.GetSize(); i++) a.push_back(rails_min[i]);
            for (ULONG i = 0; ok && i < rails_max.GetSize(); i++) b.push_back(rails_max[i]);
            out["rails_min"] = py::cast(a, py::return_value_policy::reference);
            out["rails_max"] = py::cast(b, py::return_value_policy::reference);
            return out;
        },
        py::arg("face"), py::arg("tol") = 1e-3, py::arg("tangency_deg") = 5.0,
        "Is the face a fillet (SmFace::IdentifyFillet with tangent rails)? dict: is_fillet, "
        "cross_section (0 circular, 1 approx. circular, 2 linear, 3/4/5 G1/G2/G3 blend, 6 other), "
        "solver (0 constant radius, 1 constant distance, 2 variable radius), rail_dir ('U'/'V'), radius, "
        "dist, radius_min, radius_max, rails_min, rails_max (the rail edges on either side).");

    ext.def("delete_fillet_face",
        [](SmBrep* brep, SmFace* face, bool stitch) {
            if (face == nullptr || face->GetBrep() != brep)
                throw std::invalid_argument("face does not belong to this brep");
            SmTemporaryChangeValue<SmBoolean> editing(brep->m_bEditingEnabled, TRUE);
            CHECK_STATUS(brep->DeleteFilletFace(face));
            if (stitch)
                CHECK_STATUS(brep->StitchAndOrient());
            return brep;
        },
        py::return_value_policy::reference, py::arg("brep"), py::arg("face"), py::arg("stitch") = true,
        "Remove one fillet face in place: its neighbours run on to meet in a sharp edge "
        "(SmBrep::DeleteFilletFace). Face handles are stale afterwards; on failure the brep may be "
        "half edited, so call it on a copy.");

    ext.def("defeature_faces",
        [](SmBrep* brep, std::vector<std::vector<SmFace*>>& groups, double ext_dist) {
            const SmContext& ctx = *brep->GetContext();
            SmTArray<SmFace*> orig;
            brep->GetFaces(orig);
            // the work copy keeps the face order: the groups are found in it by index
            SmBrep* work = new (ctx) SmBrep(*brep, NULL, NULL);
            SmObjDelete clean_work(work);
            SmTArray<SmFace*> wf;
            work->GetFaces(wf);
            if (wf.GetSize() != orig.GetSize())
                throw std::runtime_error("the brep copy has a different face count");
            SmFeatureExecutive exec(ctx, *work);
            std::vector<SmFeature*> feats;
            for (auto& g : groups)
            {
                SmTArray<SmFace*> grp;
                for (SmFace* f : g)
                {
                    long idx = -1;
                    for (ULONG i = 0; i < orig.GetSize(); i++)
                        if (orig[i] == f) { idx = (long)i; break; }
                    if (idx < 0)
                        throw std::invalid_argument("face does not belong to this brep");
                    grp.AddUnique(wf[idx]);
                }
                if (grp.GetSize() == 0)
                    continue;
                SmFeature* ft = exec.AddFeature(grp, SM_RB_EXTEND_SURFACE, SM_RB_FROM_OTHER_BREP);
                if (ft == nullptr)
                    throw std::runtime_error("the faces do not make a feature (a group must be connected "
                                             "and leave the body round it)");
                feats.push_back(ft);
            }
            if (feats.empty())
                return brep;
            if (ext_dist > 0.0)
                exec.SetBaseExtDist(ext_dist);
            SmBrep* base = NULL;
            SmTArray<SmBrep*> feature_breps;
            SmObjsDelete<SmBrep*> clean_features(&feature_breps);
            CHECK_STATUS(exec.DoDefeature(base, &feature_breps));
            for (SmFeature* ft : feats)
                if (!ft->GetBaseWasApplied())
                    throw std::runtime_error("the faces round a group could not be extended to close it");
            {
                SmTemporaryChangeValue<SmBoolean> editing(work->m_bEditingEnabled, TRUE);
                CHECK_STATUS(work->RemoveTopologicalEdgesAndVertices());   // rejoin faces the extensions split
            }
            if (!work->AssertValid(NULL, SM_LEVEL_2))
                throw std::runtime_error("the result is not a valid brep");
            if (brep->IsManifoldSolid() && !work->IsManifoldSolid())
                throw std::runtime_error("the result is not a closed solid");
            SmBrep* backup = new (ctx) SmBrep(*brep, NULL, NULL);
            SmObjDelete clean_backup(backup);
            SmTemporaryChangeValue<SmBoolean> editing(brep->m_bEditingEnabled, TRUE);
            SmStatus st = brep->RemoveAllTopology();
            if (st == SM_SUCCESS)
                st = brep->MergeBrep(*work);
            if (st != SM_SUCCESS && brep->RemoveAllTopology() == SM_SUCCESS)
                brep->MergeBrep(*backup);
            CHECK_STATUS(st);
            return brep;
        },
        py::return_value_policy::reference, py::arg("brep"), py::arg("groups"), py::arg("ext_dist") = 0.0,
        "Remove each group of connected faces (fillets with their corner patches, a hole, a boss, a "
        "chamfer) and close the gap by extending the faces round it (SmFeatureExecutive, "
        "SM_RB_EXTEND_SURFACE). Works on a copy: the brep is replaced only by a valid result, else left "
        "as it was and RuntimeError raised. ext_dist: how far the surfaces are extended (0: estimated "
        "from the loop's size). Face handles are stale afterwards.");

    py::class_<KKTFactor>(ext, "KKTFactor",
        "The system [[M, C^T], [C, -delta I]] factorised once (supernodal LDL^T from Accelerate on macOS, "
        "Eigen's simplicial LDL^T elsewhere); solve(rx, ry) returns x, the move of least x^T M x - 2 rx^T x "
        "meeting C x = ry, for any number of right sides.")
        .def(py::init([](const I64& mr, const I64& mc, const F64& mv, std::int64_t n,
                         const I64& cr, const I64& cc, const F64& cv, std::int64_t m, double delta, bool eigen_only) {
                 SpMat M = sparse_from(mr, mc, mv, n, n);
                 SpMat C = sparse_from(cr, cc, cv, m, n);
                 py::gil_scoped_release nogil;
                 return std::make_unique<KKTFactor>(M, C, delta, eigen_only);
             }),
             py::arg("mr"), py::arg("mc"), py::arg("mv"), py::arg("n"),
             py::arg("cr"), py::arg("cc"), py::arg("cv"), py::arg("m"),
             py::arg("delta") = 1e-12, py::arg("eigen_only") = false)
        .def("solve",
             [](const KKTFactor& k, const F64& rx, const F64& ry, int refine) {
                 if (rx.size() != k.n() || ry.size() != k.m())
                     throw std::invalid_argument("rx must have n entries, ry m");
                 Eigen::VectorXd b(k.n() + k.m());
                 std::copy(rx.data(), rx.data() + k.n(), b.data());
                 std::copy(ry.data(), ry.data() + k.m(), b.data() + k.n());
                 Eigen::VectorXd x;
                 {
                     py::gil_scoped_release nogil;
                     x = k.solve(b, refine).head(k.n());
                 }
                 return to_numpy(x);
             },
             py::arg("rx"), py::arg("ry"), py::arg("refine") = 3)
        .def_property_readonly("accelerate", &KKTFactor::accelerate)
        .def_property_readonly("times", &KKTFactor::times);

    ext.def("sparse_kkt",
        [](const I64& mr, const I64& mc, const F64& mv, std::int64_t n,
           const I64& cr, const I64& cc, const F64& cv, std::int64_t m,
           const F64& rx, const F64& ry, double delta, bool eigen_only) {
            if (rx.size() != n || ry.size() != m)
                throw std::invalid_argument("rx must have n entries, ry m");
            SpMat M = sparse_from(mr, mc, mv, n, n);
            SpMat C = sparse_from(cr, cc, cv, m, n);
            Eigen::VectorXd x;
            {
                py::gil_scoped_release nogil;
                KKTFactor k(M, C, delta, eigen_only);
                Eigen::VectorXd b(n + m);
                std::copy(rx.data(), rx.data() + n, b.data());
                std::copy(ry.data(), ry.data() + m, b.data() + n);
                x = k.solve(b).head(n);
            }
            return to_numpy(x);
        },
        py::arg("mr"), py::arg("mc"), py::arg("mv"), py::arg("n"),
        py::arg("cr"), py::arg("cc"), py::arg("cv"), py::arg("m"),
        py::arg("rx"), py::arg("ry"), py::arg("delta") = 1e-12, py::arg("eigen_only") = false,
        "Solve the quasi-definite system [[M, C^T], [C, -delta I]] [x; y] = [rx; ry] once; "
        "with rx = 0 that is the move x of least x^T M x meeting C x = ry. Returns x.");

}
