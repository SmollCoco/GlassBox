import unittest
import numpy as np
import os
import tempfile

from GlassBox.numpandas.core.dataframe import DataFrame

from GlassBox.eda.stats import UnivariateStats, IQR_OutlierDetector
from GlassBox.eda.profiler import DataProfiler
from GlassBox.eda.plotter import plot_manager

class TestEdaStats(unittest.TestCase):
    def setUp(self):
        self.arr = np.array([1, 2, 2, 3, 4, 5, 20])
        
    def test_mean(self):
        self.assertAlmostEqual(UnivariateStats.calc_mean(self.arr), np.mean(self.arr))
        
    def test_median(self):
        self.assertAlmostEqual(UnivariateStats.calc_median(self.arr), np.median(self.arr))
        
    def test_mode(self):
        self.assertEqual(UnivariateStats.calc_mode(self.arr), 2)
        
    def test_std(self):
        self.assertAlmostEqual(UnivariateStats.calc_std(self.arr), np.std(self.arr, ddof=1))
        
    def test_iqr_outlier(self):
        df = DataFrame({"a": self.arr})
        detector = IQR_OutlierDetector()
        report = detector.fit(df).get_outlier_report(df)
        self.assertEqual(report["a"], 1) # 20 is an outlier
        
        capped_df = detector.cap_outliers(df)
        self.assertLess(capped_df["a"].to_numpy()[-1], 20)

class TestProfiler(unittest.TestCase):
    def test_profiler(self):
        df = DataFrame({
            "num": [1, 2, 3, np.nan, 5],
            "cat": ["a", "b", "a", "c", "a"],
            "bools": [True, False, True, True, False]
        })
        profiler = DataProfiler(df)
        profiler.compute_profile()
        
        self.assertEqual(profiler.feature_types["num"], "Numerical")
        self.assertEqual(profiler.feature_types["cat"], "Categorical")
        self.assertEqual(profiler.feature_types["bools"], "Boolean")
        
        # Test report generation
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_report.html")
            profiler.generate_html_report(filepath)
            self.assertTrue(os.path.exists(filepath))

class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame({
            "num1": [1, 2, 3, 4, 5],
            "num2": [5, 4, 3, 2, 1],
            "cat": ["a", "b", "a", "c", "b"]
        })
        
    def test_plotters_run_without_error(self):
        # We are just testing that they don't break, actual plot asserting is complex
        try:
            # We mock plt.show to prevent window blocking during automated testing
            import matplotlib.pyplot as plt
            original_show = plt.show
            plt.show = lambda: None
            
            plot_manager.histplot(self.df, "num1")
            plot_manager.count_plotter = None # not necessary to break abstraction
            plot_manager.countplot(self.df, "cat")
            plot_manager.correlation_matrix(self.df)
            # Cannot easily test pairplot here due to seaborn's figure overriding but we covered it manually
            
            plt.show = original_show
        except Exception as e:
            self.fail(f"PlotManager raised exception {e}")

if __name__ == "__main__":
    unittest.main()
