#!/usr/bin/env python3
"""
Test script to verify the cattle breed classification setup
"""

import os
import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    
    required_packages = [
        'flask',
        'tensorflow',
        'numpy',
        'opencv-python',
        'PIL',
        'sklearn',
        'werkzeug'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nMissing packages: {', '.join(failed_imports)}")
        print("Install them with: pip install " + " ".join(failed_imports))
        return False
    
    print("All packages imported successfully!")
    return True

def test_dataset_structure():
    """Test if dataset is properly organized"""
    print("\nTesting dataset structure...")
    
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset directory '{dataset_path}' not found")
        return False
    
    expected_folders = [
        'Cross_bred-Dairy_Cattle',
        'Exotic dairy breeds of cattle',
        'Indigenous_dairy_breeds_of_cattle',
        'Indigenous_Draught_breeds_of_cattle',
        'Indigenous_Dual_purpose_breeds_of_Cattle'
    ]
    
    missing_folders = []
    total_images = 0
    
    for folder in expected_folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            # Count images in folder
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            total_images += len(images)
            print(f"✓ {folder} ({len(images)} images)")
        else:
            print(f"✗ {folder} - NOT FOUND")
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"\nMissing folders: {', '.join(missing_folders)}")
        return False
    
    print(f"\nTotal images found: {total_images}")
    if total_images < 5:
        print("⚠ Warning: Very few images found. Consider adding more images for better model performance.")
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'run_app.py',
        'requirements.txt',
        'frontend/templates/base.html',
        'frontend/templates/register.html',
        'frontend/templates/login.html',
        'frontend/templates/dashboard.html',
        'frontend/templates/result.html',
        'model/create_simple_model.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False
    
    return True

def test_flask_app():
    """Test if Flask app can be imported"""
    print("\nTesting Flask app import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the app
        from app import app
        print("✓ Flask app imported successfully")
        
        # Test if app has required routes
        required_routes = ['/', '/register', '/login', '/dashboard', '/upload']
        with app.test_client() as client:
            for route in required_routes:
                response = client.get(route)
                if response.status_code in [200, 302, 405]:  # 405 is OK for POST-only routes
                    print(f"✓ Route {route} accessible")
                else:
                    print(f"✗ Route {route} returned status {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Cattle Breed Classification - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Dataset Structure", test_dataset_structure),
        ("File Structure", test_file_structure),
        ("Flask App", test_flask_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("Run 'python run_app.py' to start the application.")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before running the application.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


