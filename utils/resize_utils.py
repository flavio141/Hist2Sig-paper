import gc
import pyvips


def resize_wsi_to_20x(resolution, mpp_x, mpp_y, input_path, output_path, image=None, resized=None):
    try:
        image = pyvips.Image.new_from_file(input_path, access="sequential")

        # 1. Check if the image has 40x or if it is 20x with wrong MPP
        if resolution == 40 or (resolution == 20 and abs(mpp_x - 0.25) < 0.03 and abs(mpp_y - 0.25) < 0.03):
            scale = 0.5
            resized = image.resize(scale)
        # 2. Check if the image has 20x with correct MPP
        else:
            resized = image

        resized.tiffsave(
            output_path,
            tile=True,
            pyramid=True,
            compression='jpeg',
            Q=90,
            bigtiff=True,
            tile_width=256,
            tile_height=256
        )

        return True

    except Exception as e:
        return False
    
    finally:
        if resized is not None:
            del resized
        if image is not None:
            del image
        
        gc.collect()
        
        try:
            pyvips.cache_set_max(0)
            pyvips.cache_set_max(100)
        except:
            pass


def force_memory_cleanup():
    gc.collect()
    try:
        pyvips.cache_drop_all()
    except:
        pass