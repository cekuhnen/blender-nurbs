// Blender NURBS add-on extension to the USD BRep Python bindings.
//
// This file replaces SmPyUsd.cpp in our macOS build (no OpenUSD there). It
// defines bind_usd() so the upstream SmPyMain.cpp links unchanged, and uses
// it to register a `blender_ext` submodule with the few kernel entry points
// the upstream Python API does not expose:
//   - build a B-spline surface / curve from explicit CVs, knots and weights
//   - read CVs, knots and weights back from curves and surfaces
//   - wrap a surface into a single-face Brep (for tessellation / export)

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
#include <algorithm>

#include <cmath>
#include <vector>

namespace
{

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
}
