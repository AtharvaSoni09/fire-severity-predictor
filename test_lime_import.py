import sys
try:
    import lime
    print("Import lime successful")
except ImportError as e:
    print(f"Import lime failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
