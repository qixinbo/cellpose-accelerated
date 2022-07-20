import os, glob
from natsort import natsorted
import cv2
import tifffile
import logging
io_logger = logging.getLogger(__name__)

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]['shape']
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                io_logger.info(f'reading tiff with {ltif} planes')
                img = np.zeros((ltif, *shape), dtype=dtype)
                for i,page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)            
        return img
    elif ext != '.npy':
        try:
            img = cv2.imread(filename, -1)#cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            io_logger.critical('ERROR: could not read file, %s'%e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat['masks']
            return masks
        except Exception as e:
            io_logger.critical('ERROR: could not read masks from file, %s'%e)
            return None

def get_image_files(folder, mask_filter, imf=None, look_one_level_down=False):
    """ find all images in a folder and if look_one_level_down all subfolders """
    mask_filters = ['_cp_masks', '_cp_output', '_flows', '_masks', mask_filter]
    image_names = []
    if imf is None:
        imf = ''
    
    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*/")))
    folders.append(folder)

    for folder in folders:
        image_names.extend(glob.glob(folder + '/*%s.png'%imf))
        image_names.extend(glob.glob(folder + '/*%s.jpg'%imf))
        image_names.extend(glob.glob(folder + '/*%s.jpeg'%imf))
        image_names.extend(glob.glob(folder + '/*%s.tif'%imf))
        image_names.extend(glob.glob(folder + '/*%s.tiff'%imf))
    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and imfile[-len(mask_filter):] != mask_filter) or len(imfile) <= len(mask_filter) 
                        for mask_filter in mask_filters])
        if len(imf)>0:
            igood &= imfile[-len(imf):]==imf
        if igood:
            imn.append(im)
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names

def get_label_files(image_names, mask_filter, imf=None):
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    print("label_names0 = ", label_names0)
    print("label_names = ", label_names)
        
    # check for flows
    if os.path.exists(label_names0[0] + '_flows.tif'):
        flow_names = [label_names0[n] + '_flows.tif' for n in range(nimg)]
    else:
        flow_names = [label_names[n] + '_flows.tif' for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        io_logger.info('not all flows are present, running flow generation for all images')
        flow_names = None
    
    # check for masks
    if mask_filter =='_seg.npy':
        label_names = [label_names[n] + mask_filter for n in range(nimg)]
        return label_names, None

    if os.path.exists(label_names[0] + mask_filter + '.tif'):
        label_names = [label_names[n] + mask_filter + '.tif' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.tiff'):
        label_names = [label_names[n] + mask_filter + '.tiff' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.png'):
        label_names = [label_names[n] + mask_filter + '.png' for n in range(nimg)]
    # todo, allow _seg.npy
    #elif os.path.exists(label_names[0] + '_seg.npy'):
    #    io_logger.info('labels found as _seg.npy files, converting to tif')
    else:
        raise ValueError('labels not provided with correct --mask_filter')
    if not all([os.path.exists(label) for label in label_names]):
        raise ValueError('labels not provided for all images in train and/or test set')

    return label_names, flow_names