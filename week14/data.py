from dipy.data import get_fnames, load_nifti

print("Starting to load data...")
# 获取数据文件名
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name='stanford_hardi')
print(f"Data file path: {hardi_fname}")

# 加载数据
data, affine = load_nifti(hardi_fname)
print(f"Data shape: {data.shape}")
