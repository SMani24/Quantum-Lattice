"""Clearly synthetic report fixtures for portable figure export and guards."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from quantum_lattice.models.final_plots import render_final_plots


def write_fixture(run):
    """Write table-only test data; no simulated or research results are claimed."""
    run=Path(run)
    run.mkdir(parents=True,exist_ok=True)
    sizes=[3,4,8,12,16,20,24]
    seeds=[101,102,103,104,105]
    arms=[("map_mlp","f"),("cnn","f"),("map_mlp","g"),("cnn","g")]
    plan={"synthetic_fixture":True,"config":{"sizes":sizes,"seeds":seeds,
        "confirmation":{"sampling":{"total_sweeps":100000,"burn_in_sweeps":10000,"thinning_sweeps":20}}}}
    (run/"plan.json").write_text(json.dumps(plan))
    rows=[]
    means={}
    for role in ("validation","confirmation"):
        for L in sizes:
            for arm_index,(model,view) in enumerate(arms):
                for i,seed in enumerate(seeds):
                    mse=.08+.0005*L-.008*arm_index+.001*(i-2)+(.002 if role=="confirmation" else 0)
                    rows.append({"role":role,"L":L,"seed":seed,"model":model,"view":view,
                        "key":f"L{L}-seed{seed}-{model}-{view}","mse":mse,
                        "D5_reference_sign":.3+.001*L-.025*arm_index+.002*(i-2),"hamiltonians":8})
                    means.setdefault((role,L,model,view),[]).append(mse)
    with (run/"model-scores.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    contrasts=[]
    for L in sizes:
        for label,left,right in [("g_minus_f_cnn",("cnn","g"),("cnn","f")),
                                 ("cnn_minus_mlp_g",("cnn","g"),("map_mlp","g"))]:
            a=means[("confirmation",L,*left)]
            b=means[("confirmation",L,*right)]
            delta=sum(a)/len(a)-sum(b)/len(b)
            contrasts.append({"role":"confirmation","L":L,"comparison":label,"metric":"mse",
                "statistical_primary":L==4 and label=="g_minus_f_cnn",
                "difference_left_minus_right":delta,"bootstrap_95ci_difference":[delta-.002,delta+.002],
                "hamiltonians":8,"optimizer_seeds":5,"bootstrap":{"confidence_level":.95}})
    (run/"paired-comparisons.json").write_text(json.dumps({"scope":"SYNTHETIC TEST FIXTURE","contrasts":contrasts}))
    return run


class FinalPlotTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix="ql-synthetic-final-plots-")
        self.addCleanup(temporary.cleanup)
        self.run=write_fixture(Path(temporary.name)/"fixture")

    def test_vector_preview_exports_scope_watermark_and_immutable_reuse(self):
        paths=render_final_plots(self.run)
        expected={"observable-performance.pdf","observable-performance.svg","observable-performance.png",
                  "paired-effects.pdf","paired-effects.svg","paired-effects.png","final-figure-gallery.pdf","plot-provenance.json"}
        self.assertEqual(set(paths),expected)
        for name,path in paths.items():
            data=Path(path).read_bytes()
            self.assertGreater(len(data),100)
            if name.endswith(".pdf"):
                self.assertTrue(data.startswith(b"%PDF-"))
                self.assertNotIn(b"/Subtype /Image",data)
            if name.endswith(".svg"):
                text=data.decode()
                self.assertIn("SYNTHETIC TEST FIXTURE",text)
                self.assertIn("Known-field",text)
        metadata=json.loads(Path(paths["plot-provenance.json"]).read_text())
        self.assertTrue(metadata["synthetic_fixture"])
        self.assertTrue(metadata["text_bounds_checked"])
        before={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in paths.items()}
        self.assertEqual(render_final_plots(self.run),paths)
        self.assertEqual(before,{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in paths.items()})
        with Path(paths["paired-effects.svg"]).open("ab") as handle:handle.write(b"<!-- altered -->")
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):
            render_final_plots(self.run)

    def test_incomplete_grid_and_wrong_primary_are_refused(self):
        path=self.run/"model-scores.csv"
        lines=path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1])+"\n")
        with self.assertRaisesRegex(ValueError,"complete matched"):
            render_final_plots(self.run)
        write_fixture(self.run)
        path=self.run/"paired-comparisons.json"
        data=json.loads(path.read_text());data["contrasts"][0]["statistical_primary"]=True
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"scope/count metadata"):
            render_final_plots(self.run)


if __name__=="__main__":unittest.main()
