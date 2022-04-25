import argparse
import cv2
import numpy as np
import os
import SimpleITK as sitk
import torch
from torch import nn
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='D:\\ESMIRA\\ESMIRA_2D_v1\\train\\CSA',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam++',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    # model = models.resnet50(pretrained=True)  # 选择指定的层 vgg中应写作：model = models.vgg16(pretrained=True)
    # 如果是指定模型应该是：
    model = models.vgg16(pretrained=False)
    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, 2)
    parameters = model.parameters()
    resume_path = 'C:\\Users\yli5\\PycharmProjects\\M4E_2d_2class\\checkpoints\\5fold_eac_atl_1200\\vgg16_16_fold1best_model.pth.tar'
    if os.path.isfile(resume_path):
        # print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    # target_layers = [model.layer4[-1]]
    target_layers = [model.features[-1]]  # 取最后一层CNN的最后一层

    # ————————————————————————————以下为输入输出设置，进入函数循环—————————————————————————————————#
    img_list = os.listdir(args.image_path)
    for ori_img in img_list[:600]:
        if 'slice9' in ori_img:
            img_path = os.path.join(args.image_path, ori_img)
            img_name = str(ori_img[:-4])
            # 采用：：-1的方式颠倒了数组的排布
            # 用医学图像则使用 sitk读进来转成np数组，然后复制三份的方式
            gray_img = sitk.ReadImage(img_path)
            array_img = sitk.GetArrayFromImage(gray_img)
            data_array = np.squeeze(array_img, axis=0)
            data_array = data_array
            pixels = data_array[data_array >0]
            mean = pixels.mean()
            std = pixels.std()
            data_normal = (data_array - mean) / std
            [y, x] = data_normal.shape
            return_data = np.zeros([1, 3, y, x])
            return_data[0, 0, :] = data_normal
            return_data[0, 1, :] = data_normal
            return_data[0, 2, :] = data_normal
            # print(return_data.shape)
            input_tensor = torch.from_numpy(return_data).to(torch.float32)

            # create a [0, 1] range data img
            data_img = data_normal.astype(np.float32)
            data_img_max = np.max(data_img)
            data_img_min = np.min(data_img)
            data_normal_img = (data_img - data_img_min) / (data_img_max - data_img_min)
            img = np.zeros([3, 512, 512])
            img[0, :] = data_normal_img
            img[1, :] = data_normal_img
            img[2, :] = data_normal_img
            img = img.transpose(1, 2, 0)
            # print(img.shape)


            # rgb_img = np.float32(rgb_img) / 255
            # 医学图像中改成归一化
            # calculate the mean and std
            # pixels = img[img > 0]
            # mean = pixels.mean()
            # std = pixels.std()
            # input_tensor = preprocess_image(img, mean=mean, std=std)
            # input_tensor = preprocess_image(rgb_img,
            #                                 mean=[0.485, 0.456, 0.406],
            #                                 std=[0.229, 0.224, 0.225])

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = 1


            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            cam_algorithm = methods[args.method]
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32

                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]

                cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
            gb = gb_model(input_tensor, target_category=target_category)

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)

            # cv2.imwrite(f'./result/{img_name}_cam.jpg', cam_image)
            # cv2.imwrite(f'./result/{img_name}_gb.jpg', gb)
            # cv2.imwrite(f'./result/{img_name}_cam_gb.jpg', cam_gb)
            # cv2.imwrite(f'./result/{img_name}_original.jpg', data_array)
            label = np.argmax(model(input_tensor).cpu().data.numpy())
            folder_output = '086_alleac_result'
            target_output = 'csa'

            cv2.imwrite(f'D:/ESMIRA/ESMIRA_2D_v1/{folder_output}/cam_{target_output}_result/{label}_{img_name[3:]}_cam.jpg', cam_image)
            # cv2.imwrite(f'D:/ESMIRA/ESMIRA_2D_v1/{folder_output}/cam_{target_output}_result/{label}_{img_name[3:]}_gb.jpg', gb)
            # cv2.imwrite(f'D:/ESMIRA/ESMIRA_2D_v1/{folder_output}/cam_{target_output}_result/{label}_{img_name[3:]}_cam_gb.jpg', cam_gb)
            cv2.imwrite(f'D:/ESMIRA/ESMIRA_2D_v1/{folder_output}/cam_{target_output}_result/{label}_{img_name[3:]}_original.jpg', data_array)
        else:
            continue
