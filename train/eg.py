from patch_loader import create_patch_loader

BATCH_SIZE = 32
# 加载 train / val / test 数据
train_loader = create_patch_loader("data/SOLAR/patches/solar_train.npz", batch_size=BATCH_SIZE, shuffle=True)
val_loader   = create_patch_loader("data/SOLAR/patches/solar_val.npz", batch_size=BATCH_SIZE, shuffle=False)
test_loader  = create_patch_loader("data/SOLAR/patches/solar_test.npz", batch_size=BATCH_SIZE, shuffle=False)

# 迭代读取一个 batch
for x_batch, y_batch in train_loader:
    print(x_batch.shape)  # torch.Size([32, 11, 16, 137])
    print(y_batch.shape)  # torch.Size([32, 11, 16, 137])
    #print how many x y pairs in train_loader
    print(len(train_loader))
    break
