"""Stdlib receipt checks for the preregistered E-03 surrogate-LOCO procedure."""
import copy
import unittest

from experiments.e03_tuning import validate_hyperparameters


class HyperparameterReceiptTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"gamma_factor": .01}, {"gamma_factor": 1.}]
        self.grid = {"gamma_factor": [.01, 1.]}
        self.folds = [dict(surrogate_family=family, fold=0, n_train=20, n_known=10, n_surrogate=10)
                      for family in ("Blip", "Tomte")]

        def calibration(factor):
            return dict(target_fpr=.05, observed_fpr=.05, median_squared_distance=2.,
                        effective_gamma=factor/2., nu_trials=[dict(nu=nu, fpr=nu)
                        for nu in (.01, .05, .1, .15, .2)])

        rows = []
        for params, value in zip(self.candidates, (.6, .9)):
            rows.append(dict(params=params, mean_surrogate_auroc=value, folds=[fold | dict(
                auroc=value, calibrated_params=params | {"nu": .05},
                calibration=calibration(params["gamma_factor"])) for fold in self.folds]))
        self.report = dict(tuning_strategy="nested_surrogate_loco_auroc", n_configs_evaluated=2,
            inner_cv=rows, best_params={"gamma_factor": 1., "nu": .05}, best_cv_score=.9,
            selection_audit=dict(grid=self.grid, objective="mean_surrogate_auroc", inner_splits=2,
                                 best_candidate=1, final_nu_calibration=calibration(1.)))

    def validate(self):
        return validate_hyperparameters(self.report, self.candidates, self.grid, self.folds)

    def test_valid_surrogate_discrimination_receipt(self):
        self.assertEqual(self.validate()["best_candidate"], 1)

    def test_fpr_only_selection_cannot_validate_H3(self):
        self.report["tuning_strategy"] = "unlabelled_calibration_only_not_H3"
        with self.assertRaisesRegex(ValueError, "H3 requires"):
            self.validate()

    def test_wrong_candidate_mean_is_rejected(self):
        self.report["inner_cv"][0]["mean_surrogate_auroc"] = .8
        with self.assertRaisesRegex(ValueError, "mean does not reproduce"):
            self.validate()

    def test_wrong_best_gamma_is_rejected(self):
        self.report["best_params"]["gamma_factor"] = .01
        with self.assertRaisesRegex(ValueError, "selected hyperparameters"):
            self.validate()

    def test_wrong_gamma_distance_normalization_is_rejected(self):
        self.report["inner_cv"][0]["folds"][0]["calibration"]["effective_gamma"] = 100.
        with self.assertRaisesRegex(ValueError, "median squared"):
            self.validate()

    def test_wrong_nu_target_is_rejected(self):
        self.report["selection_audit"]["final_nu_calibration"]["target_fpr"] = .01
        with self.assertRaisesRegex(ValueError, "target 5%"):
            self.validate()

    def test_wrong_nu_selection_is_rejected(self):
        self.report["best_params"]["nu"] = .2
        with self.assertRaisesRegex(ValueError, "nu was not selected"):
            self.validate()

    def test_missing_surrogate_fold_is_rejected(self):
        self.report["inner_cv"][0]["folds"].pop()
        with self.assertRaisesRegex(ValueError, "fold coverage"):
            self.validate()

    def test_wrong_surrogate_family_is_rejected(self):
        self.report["inner_cv"][0]["folds"][0]["surrogate_family"] = "Fast_Scattering"
        with self.assertRaisesRegex(ValueError, "fold membership"):
            self.validate()

    def test_nonfinite_surrogate_score_is_rejected(self):
        self.report["inner_cv"][0]["folds"][0]["auroc"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid surrogate"):
            self.validate()

    def test_equal_auroc_prefers_smaller_gamma(self):
        self.report["inner_cv"][1] = copy.deepcopy(self.report["inner_cv"][0])
        self.report["inner_cv"][1]["params"] = self.candidates[1]
        for fold in self.report["inner_cv"][1]["folds"]:
            fold["calibrated_params"]["gamma_factor"] = 1.
            fold["calibration"]["effective_gamma"] = .5
        self.report["best_params"] = {"gamma_factor": .01, "nu": .05}
        self.report["best_cv_score"] = .6
        self.report["selection_audit"]["best_candidate"] = 0
        self.report["selection_audit"]["final_nu_calibration"]["effective_gamma"] = .005
        self.assertEqual(self.validate()["best_candidate"], 0)


if __name__ == "__main__":
    unittest.main()
