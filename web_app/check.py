import os

test_dir = r"C:\Haswin info\Medical imaging project\AI_In_Radiography\X_ray-dataset\chest_xray\test"

# Check if test_dir exists
if not os.path.exists(test_dir):
    print("❌ Error: Test directory does not exist!")
else:
    print("✅ Test directory exists!")
    print("📂 Subfolders:", os.listdir(test_dir))  # Print folder contents
