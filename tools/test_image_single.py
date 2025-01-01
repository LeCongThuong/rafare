import sys, os, numpy as np, time, cv2
sys.path.append("./")
from opt import opt
from engineer.core.test import recon_single, merge_norm, load_trained_model
from engineer.utils.renderer import render_rafare, align_rafare, rotate_verts_y
from engineer.utils.mmcv_config import Config
import pandas as pd
from tqdm import tqdm
from pathlib import Path


sys.path.append("engineer/face_parse/")
from face_parse import face_parse

sys.path.append("engineer/norm_pred/")
from norm_pred import norm_pred



if __name__ == "__main__":
    hg_base = load_trained_model(os.path.join('.', "configs", "SDF_FS103450_HG_base.py"), "epoch_best.tar")
    hg_fine = load_trained_model(os.path.join('.', "configs", "SDF_FS103450_HG_fine.py"), "epoch_best.tar")
    csv_file = "/mnt/hmi/thuong/Photoface_dist/PhotofaceDBNormalTrainValTest2/dataset_0/test.csv"
    data_dir =  "/mnt/hmi/thuong/Photoface_dist/PhotofaceDBLib/"
    result_dir = "./results"

    data_info = pd.read_csv(csv_file, header=None)
    num_img = len(data_info)


    for image_idx in tqdm(range(num_img)):
        im_path = os.path.join(data_dir, data_info.iloc[image_idx, 0])
    
        src_img = cv2.imread(str(im_path))
        if src_img.shape != (512, 512, 3):
            src_img = cv2.resize(src_img, (512, 512), interpolation=cv2.INTER_LINEAR)

        
        # predict semantic mask
        src_sem_mask = face_parse(src_img[:,:,::-1])
        
        # predict normal maps
        norm_front = norm_pred(src_img, front=True)
        norm_back = norm_pred(src_img, front=False)

        # recon base
        sdf_base, mat = recon_single(hg_base, src_img, src_sem_mask, num_samples=opt.num_samples)

        # recon fine
        sdf_fine, mat = recon_single(hg_fine, src_img, src_sem_mask, num_samples=opt.num_samples)

        mesh = merge_norm(sdf_base, sdf_fine, norm_front, norm_back, src_sem_mask, mat)
    
        # # save
        # os.makedirs(tgt_dir, exist_ok=True)
        # mesh.export(os.path.join(tgt_dir, src_name+'.obj'));
        
        # reload mesh
        pred_align_mesh = align_rafare(mesh, -1)

        # render in front view
        normal = render_rafare(pred_align_mesh)
        normal = normal.reshape((512, 512))
        print(normal.shape)
        cv2.resize(normal, (256, 256), interpolation=cv2.INTER_NEAREST)
        dest_path = os.path.join(result_dir, data_info.iloc[image_idx, 0]).replace("crop.jpg", "predict.npy")
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(dest_path, normal)