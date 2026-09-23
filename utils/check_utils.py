import openslide

def find_level_for_target_mpp(slide_path, target_mpp):
    slide = openslide.OpenSlide(slide_path)

    mpp_x = mpp_y = None

    try:
        x_resolution = float(slide.properties['tiff.XResolution'])
        y_resolution = float(slide.properties['tiff.YResolution'])
        resolution_unit = slide.properties.get('tiff.ResolutionUnit')

        if resolution_unit == 'centimeter':
            mpp_x = 10000 / x_resolution
            mpp_y = 10000 / y_resolution
    except (KeyError, ValueError):
        pass

    if mpp_x is None or mpp_y is None:
        try:
            mpp_x = float(slide.properties['openslide.mpp-x'])
            mpp_y = float(slide.properties['openslide.mpp-y'])
        except (KeyError, ValueError):
            return None, None, None, None, slide.properties

    objective_power = (
        slide.properties.get('aperio.AppMag')
        or slide.properties.get('openslide.objective-power')
    )
    try:
        objective_power = int(float(objective_power))
    except (TypeError, ValueError):
        objective_power = None

    # for level in range(slide.level_count):
    #     level_mpp_x = mpp_x * slide.level_downsamples[level]
    #     level_mpp_y = mpp_y * slide.level_downsamples[level]

    #     if abs(level_mpp_x - target_mpp) < 0.1 and abs(level_mpp_y - target_mpp) < 0.1:
    #         return level, mpp_x, mpp_y, objective_power, slide.level_downsamples, slide.properties

    # return None, mpp_x, mpp_y, objective_power, slide.level_downsamples, slide.properties
    return mpp_x, mpp_y, objective_power, slide.level_downsamples, slide.properties

