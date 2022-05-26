import os

from PIL import Image
from tqdm import tqdm

from pspnet import PSPNet
from utils.utils_metrics import compute_mIoU, show_results

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 1
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes = ["_background_", "1"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    infer_path = "test2"
    VOCdevkit_path = 'VOCdevkit/VOC2007/' + infer_path
    gt_dir = os.path.join(VOCdevkit_path, "SegmentationClass")
    imgs = os.listdir(VOCdevkit_path + "/JPEGImages")

    img_out_path = "img_out/"+infer_path
    miou_out_path = "miou_out/"+infer_path
    image_ids = []
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(img_out_path):
            os.makedirs((img_out_path))
        if not os.path.exists(miou_out_path):
            os.makedirs((miou_out_path))

        print("Load model.")
        pspnet = PSPNet()
        print("Load model done.")

        print("Get predict result.")

        for img in tqdm(imgs):
            image_path = os.path.join(VOCdevkit_path+ "/JPEGImages", img)
            image = Image.open(image_path)
            #get Iou
            image_iou = pspnet.get_miou_png(image)
            image_iou.save(os.path.join(miou_out_path, img[:-4] + ".png"))
            image_ids.append(img[:-4])
            # infenrence
            ima_det = pspnet.detect_image(image)
            ima_det.save(os.path.join(img_out_path, img[:-4] + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, miou_out_path, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)