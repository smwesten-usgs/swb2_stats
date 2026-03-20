# tests/test_imports.py
def test_can_import_package():
    import swb2_stats
    assert hasattr(swb2_stats, "create_summary_dataset")
    assert hasattr(swb2_stats, "calculate_zonal_statistics")
