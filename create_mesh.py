#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script performs segmentation and surface extraction. The segmentation is based on watershed algorithm or
global threshold.
"""

import argparse
import numpy as np
import time
import SimpleITK as sitk
from utils import sitk2vtk
from utils import dicomutils
from utils import vtkutils

__author__ = "ITO Tsuyoshi"
__version__ = "0.1.3"
__email__ = "ito.tsuyoshi.3a@kyoto-u.ac.jp"
__date__ = "2025-03-05"


def main():
    args = parse_arguments()

    # Read dicom files
    img, modality = dicomutils.loadLargestSeries(args.input)

    # Modify orientation
    img_direction = img.GetDirection()
    if img_direction[0] == -1:
        t = time.perf_counter()
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        print("Direction modified")
        print("    Original image direction: ", img_direction)
        print("    Modified image direction: ", img.GetDirection())
        vtkutils.elapsedTime(t)

    # Write image as NIfTi format
    if args.save_volume:
        sitk.WriteImage(img, args.output + ".nii")

    # Resampling
    if args.resample_cubic:
        img = resample_img(img, spacing=img.GetSpacing()[2])

    elif args.resample_img is not None:
        img = resample_img(img, spacing=args.resample_img)

    # Smoothing using curvature flow filter
    if args.smooth_img:
        img = smooth_img(img, time_step=args.cf_step, iterations=args.cf_iterations)

    # Segmentation: watershed or global threshold
    if args.segmentation == "watershed":
        img_segmented = seg_watershed(img,
                                      mask_value=args.mask_value,
                                      bone_lower=args.bone_lower,
                                      air_upper=args.air_upper,
                                      seeds_bone=args.seeds_bone)
    else:
        t = time.perf_counter()
        img_segmented = img > args.threshold
        print("Image segmented")
        print("    ", np.count_nonzero(sitk.GetArrayFromImage(img_segmented) == 1), "voxels")
        vtkutils.elapsedTime(t)

    # Convert sitk to vtk
    vtk_img = sitk2vtk.sitk2vtk(img_segmented)

    # Extract surface
    mesh = vtkutils.extractSurface(vtk_img, 0.5)

    # Clean mesh
    mesh = vtkutils.cleanMesh(mesh)

    # Reduce mesh
    if args.reduce_mesh is not None:
        mesh = vtkutils.reduceMesh(mesh, args.reduce_mesh)
    
    # To avoid keeping normals in newer versions of vtk 
    mesh.GetPointData().SetNormals(None)

    # Smooth mesh
    if args.smooth_mesh is not None:
        mesh = vtkutils.smoothMesh(mesh, args.smooth_mesh)

    # Write mesh as ply format
    vtkutils.writeMesh(mesh, args.output + ".ply")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-i",
                        "--input",
                        action="store",
                        dest="input",
                        required=True,
                        help="Path to the input directory")

    parser.add_argument("-o",
                        "--output",
                        action="store",
                        dest="output",
                        required=True,
                        help="Path to the output directory and prefix")

    parser.add_argument("-r",
                        "--resample_img",
                        action="store",
                        dest="resample_img",
                        default=None,
                        type=float,
                        help="Spacing for resampling")

    parser.add_argument("-v",
                        "--save_volume",
                        action="store_true",
                        dest="save_volume",
                        help="Save volume as NIfTi format or not")

    parser.add_argument("-c",
                        "--resample_cubic",
                        action="store_true",
                        dest="resample_cubic",
                        help="Resample to be cubic or not")

    parser.add_argument("-s",
                        "--smooth_img",
                        action="store_true",
                        dest="smooth_img",
                        help="Perform smoothing (curvature flow filter) or not")

    parser.add_argument("--cf_step",
                        action="store",
                        dest="cf_step",
                        default=0.005,
                        type=float,
                        help="Time step for curvature flow filter")

    parser.add_argument("--cf_iterations",
                        action="store",
                        dest="cf_iterations",
                        default=5,
                        type=int,
                        help="Iterations for curvature flow filter")

    parser.add_argument("-g",
                        "--segmentation",
                        action="store",
                        dest="segmentation",
                        default="watershed",
                        help="Segmentation method: watershed or threshold")

    parser.add_argument("--threshold",
                        action="store",
                        dest="threshold",
                        default=0,
                        type=float,
                        help="Global threshold for segmentation")

    parser.add_argument("--mask_value",
                        action="store",
                        dest="mask_value",
                        default=-700,
                        type=float,
                        help="The voxels under this value are not used for gradient calculation")

    parser.add_argument("--bone_lower",
                        action="store",
                        dest="bone_lower",
                        default=-500,
                        type=float,
                        help="Lower bound for bone")

    parser.add_argument("--air_upper",
                        action="store",
                        dest="air_upper",
                        default=-1000,
                        type=float,
                        help="Upper bound for air")

    parser.add_argument("--seeds_bone",
                        action="store",
                        dest="seeds_bone",
                        default=0,
                        type=float,
                        help="Seeds for bone")

    parser.add_argument("-d",
                        "--reduce_mesh",
                        action="store",
                        dest="reduce_mesh",
                        default=None,
                        type=float,
                        help="Reduction factor for reducing mesh")

    parser.add_argument("-m",
                        "--smooth_mesh",
                        action="store",
                        dest="smooth_mesh",
                        default=None,
                        type=int,
                        help="Iterations for smoothing mesh")

    args = parser.parse_args()
    return args


def resample_img(img=None, spacing=None, interporator=sitk.sitkLinear, defaultPixelValue=-2048):
    t = time.perf_counter()
    identity = sitk.Transform(3, sitk.sitkIdentity)

    x_size = int(img.GetSize()[0] * (img.GetSpacing()[0] / spacing))
    y_size = int(img.GetSize()[1] * (img.GetSpacing()[1] / spacing))
    z_size = int(img.GetSize()[2] * (img.GetSpacing()[2] / spacing))

    sizevec = [x_size, y_size, z_size]

    img_resampled = sitk.Resample(img, sizevec, identity, interporator,
                                  img.GetOrigin(), [spacing] * 3, img.GetDirection(), defaultPixelValue)
    print("Image resampled")
    print("    Spacing: ", spacing)
    vtkutils.elapsedTime(t)

    return img_resampled


def smooth_img(img=None, time_step=None, iterations=None):
    t = time.perf_counter()
    img_smoothed = sitk.CurvatureFlow(image1=img,
                                      timeStep=time_step,
                                      numberOfIterations=iterations)
    print("Image smoothed")
    print("    ", time_step, "steps, ", iterations, "iterations")
    vtkutils.elapsedTime(t)

    return img_smoothed


def seg_watershed(img=None, mask_value=None, bone_lower=None, air_upper=None, seeds_bone=None):
    t = time.perf_counter()
    # Masking
    img_masked = sitk.Mask(img, img > mask_value, -1000, 0)

    # Sobel filter
    sobel = sitk.SobelEdgeDetection(sitk.Cast(sitk.RescaleIntensity(img_masked), sitk.sitkFloat64))

    # Seeds
    points = np.where(sitk.GetArrayFromImage(img > seeds_bone) == 1)
    seed_list = []
    for i in range(len(points[0])):
        seed_list += [(int(points[0][i]), int(points[1][i]), int(points[2][i]))]

    connected = sitk.ConnectedThreshold(img, seedList=seed_list, lower=bone_lower, upper=10000)

    seeds = sitk.BinaryErode(connected)

    air = sitk.BinaryThreshold(img, lowerThreshold=-10000, upperThreshold=air_upper, insideValue=2, outsideValue=0)

    # watershed
    markers = seeds + air
    ws = sitk.MorphologicalWatershedFromMarkers(sobel, markers, markWatershedLine=False, fullyConnected=False)
    img_segmented = ws < 2  # remove the label of air

    print("Image segmented")
    print("    ", np.count_nonzero(sitk.GetArrayFromImage(img_segmented) == 1), "voxels")
    vtkutils.elapsedTime(t)

    return img_segmented


if __name__ == "__main__":
    main()
