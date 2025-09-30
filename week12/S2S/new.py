import tensorflow as tf
import network.Punet
import numpy as np
import util
import cv2
import os
import gc

# 配置GPU内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def train(file_path, dropout_rate, sigma=25, is_realnoisy=False):
    try:
        print(f"开始处理: {file_path}, sigma: {sigma}")
        tf.compat.v1.reset_default_graph()

        gt = util.load_np_image(file_path)
        _, w, h, c = np.shape(gt)
        model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + "/model/Self2Self/"
        os.makedirs(model_path, exist_ok=True)
        noisy = util.add_gaussian_noise(gt, model_path, sigma)
        model = network.Punet.build_denoising_unet(noisy, 1 - dropout_rate, is_realnoisy)

        loss = model['training_error']
        summary = model['summary']
        saver = model['saver']
        our_image = model['our_image']
        is_flip_lr = model['is_flip_lr']
        is_flip_ud = model['is_flip_ud']
        avg_op = model['avg_op']
        slice_avg = model['slice_avg']
        optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        avg_loss = 0
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            summary_writer = tf.compat.v1.summary.FileWriter(model_path, sess.graph)

            for step in range(N_STEP):
                feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}

                # 构建 fetches 列表，检查每个元素是否为 None
                fetches = [optimizer, avg_op, loss, our_image]
                fetch_names = ['optimizer', 'avg_op', 'loss', 'our_image']

                # 只有当 summary 不是 None 时才加入 fetches
                if summary is not None:
                    fetches.append(summary)
                    fetch_names.append('summary')

                print("fetches =", fetches)

                # 运行 session
                results = sess.run(fetches, feed_dict=feet_dict)

                # 解析结果
                if summary is not None:
                    _, _op, loss_value, o_image, merged = results
                else:
                    _, _op, loss_value, o_image = results
                    merged = None

                avg_loss += loss_value

                if (step + 1) % N_SAVE == 0:

                    print("After %d training step(s)" % (step + 1),
                          "loss  is {:.9f}".format(avg_loss / N_SAVE))
                    avg_loss = 0

                    sum = np.float32(np.zeros(our_image.shape.as_list()))
                    for j in range(N_PREDICTION):
                        feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2)}
                        o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                        sum += o_image

                    o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                    o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))

                    if is_realnoisy:
                        cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_avg)
                    else:
                        cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_image)
                    saver.save(sess, model_path + "model.ckpt-" + str(step + 1))

                # 只有当 merged 不是 None 时才写入 summary
                if merged is not None:
                    summary_writer.add_summary(merged, step)

    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 强制清理资源
        gc.collect()

    return True


if __name__ == '__main__':
    TF_DATA_TYPE = tf.float32
    LEARNING_RATE = 1e-4
    N_PREDICTION = 100
    N_SAVE = 1000
    N_STEP = 150000

    print("跳过Set9数据集，直接运行后续数据集...")

    # 跳过Set9，直接处理BSD68 - sigma=25
    path = './testsets/BSD68/'
    if not os.path.exists(path):
        print(f"错误: 数据集路径不存在: {path}")
        exit(1)

    file_list = [f for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f))]
    print(f"BSD68数据集 (sigma=25): 找到 {len(file_list)} 个文件")

    sigma = 25
    success_count = 0
    for i, file_name in enumerate(file_list):
        print(f"进度: BSD68 sigma{sigma} - {i + 1}/{len(file_list)} - {file_name}")
        success = train(os.path.join(path, file_name), 0.2, sigma)
        if success:
            success_count += 1
        else:
            print(f"文件处理失败: {file_name}")

    print(f"BSD68 sigma{sigma} 处理完成: {success_count}/{len(file_list)} 个文件成功")

    # 处理BSD68 - sigma=50
    print("开始处理BSD68数据集 (sigma=50)...")
    sigma = 50
    success_count = 0
    for i, file_name in enumerate(file_list):
        print(f"进度: BSD68 sigma{sigma} - {i + 1}/{len(file_list)} - {file_name}")
        success = train(os.path.join(path, file_name), 0.3, sigma)
        if success:
            success_count += 1
        else:
            print(f"文件处理失败: {file_name}")

    print(f"BSD68 sigma{sigma} 处理完成: {success_count}/{len(file_list)} 个文件成功")

    # 处理PolyU真实噪声数据集
    path = './testsets/PolyU/'
    if not os.path.exists(path):
        print(f"错误: 数据集路径不存在: {path}")
        exit(1)

    file_list = [f for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f))]
    print(f"PolyU真实噪声数据集: 找到 {len(file_list)} 个文件")

    sigma = -1
    success_count = 0
    for i, file_name in enumerate(file_list):
        print(f"进度: PolyU真实噪声 - {i + 1}/{len(file_list)} - {file_name}")
        success = train(os.path.join(path, file_name), 0.3, sigma, is_realnoisy=True)
        if success:
            success_count += 1
        else:
            print(f"文件处理失败: {file_name}")

    print(f"PolyU真实噪声处理完成: {success_count}/{len(file_list)} 个文件成功")
    print("所有数据集处理完成！")