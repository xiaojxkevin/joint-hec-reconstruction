import numpy as np
import cv2
import glob
import os
from scipy.spatial.transform import Rotation as R


class Calibration:
    def __init__(
        self,
        img_path: str,
        hand_path: str,
        pts_per_row: int,
        pts_per_col: int,
        scale: float,
        vis_flag=True,
    ) -> None:

        self.img_path = img_path
        self.hand_path = hand_path
        self.pts_per_row = pts_per_row
        self.pts_per_col = pts_per_col
        self.scale = scale
        self.vis_flag = vis_flag
        self.valid_ids = []

    def run(self):
        """https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b"""
        # Camera
        mtx, dist, rvecs, tvecs = self.calib_K_extrinsics()
        print("valid ids are \n", self.valid_ids)
        rvecs = rvecs[self.valid_ids]
        tvecs = tvecs[self.valid_ids]
        R_obj2cam = R.from_rotvec(rvecs.reshape((-1, 3))).as_matrix()
        t_obj2cam = tvecs.reshape((-1, 3))

        # Hand poses
        t_hand2base, R_hand2base = self.load_hand_poses()
        t_hand2base = t_hand2base[self.valid_ids]
        R_hand2base = R_hand2base[self.valid_ids]

        # calibrate hand-eye
        R_cam2hand, t_cam2hand = cv2.calibrateHandEye(
            R_gripper2base=R_hand2base,
            t_gripper2base=t_hand2base,
            R_target2cam=R_obj2cam,
            t_target2cam=t_obj2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
        T_cam2hand = np.eye(4)
        T_cam2hand[:3, :3] = R_cam2hand
        T_cam2hand[:3, 3] = t_cam2hand.reshape((-1,))
        return T_cam2hand

    def calib_K_extrinsics(self):
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.pts_per_col * self.pts_per_row, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.pts_per_row, 0 : self.pts_per_col].T.reshape(
            -1, 2
        )
        objp *= self.scale
        objpoints = []
        imgpoints = []
        images = sorted(glob.glob(self.img_path))
        for i, fname in enumerate(images):
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
            ret, corners = cv2.findChessboardCorners(
                gray, (self.pts_per_row, self.pts_per_col)
            )
            if ret:
                self.valid_ids.append(i)
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                if self.vis_flag:
                    tmp_img = cv2.drawChessboardCorners(
                        img, (self.pts_per_row, self.pts_per_col), corners2, ret
                    )
                    cv2.imshow(fname, tmp_img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(fname)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            None,
            None,
        )
        print("ret:", ret)
        print("mtx:\n", mtx)
        print("dist:\n", dist)
        # print("relative rotations:\n", rvecs)
        # print("relative translation:\n", tvecs)
        return mtx, dist, np.asarray(rvecs), np.asarray(tvecs)

    def load_hand_poses(self):
        data = np.genfromtxt(self.hand_path, dtype=np.float32)
        trans = data[:, 1:4]
        quat = data[:, 4:8]
        rots = R.from_quat(quat).as_matrix()
        return trans, rots


def main():
    # Modify this
    img_path = "data/chess_demo/imgs/*.png"
    hand_path = "data/chess_demo/hand_tum.txt"
    calib = Calibration(
        img_path, hand_path, pts_per_row=6, pts_per_col=7, scale=0.1, vis_flag=True
    )
    T_cam2hand = calib.run()
    np.savetxt("./T_cam2hand.txt", T_cam2hand, fmt="%.6f")


if __name__ == "__main__":
    main()
