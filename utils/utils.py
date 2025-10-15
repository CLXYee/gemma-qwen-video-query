import numpy as np
import ctypes as C
from jetson_utils import cudaFont

def dtype_to_ctype(dtype):
    if dtype == np.uint8:
        return C.c_uint8
    elif dtype == np.int8:
        return C.c_int8
    elif dtype == np.uint16:
        return C.c_uint16
    elif dtype == np.int16:
        return C.c_int16
    elif dtype == np.int32:
        return C.c_int32
    elif dtype == np.float32:
        return C.c_float
    elif dtype == np.float64:
        return C.c_double
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")
 
def cudaToNumpy(frame, shape=None, dtype=np.uint8):
    import numpy as np
    import ctypes as C

    # Extract the CUDA pointer from the frame
    # The field might be `ptr` or `image.ptr` depending on your version
    if hasattr(frame, "ptr"):
        ptr = frame.ptr
    else:
        raise TypeError(f"Object {type(frame)} has no 'ptr' attribute")

    if shape is None:
        shape = (frame.height, frame.width, 3)

    array = np.ctypeslib.as_array(C.cast(ptr, C.POINTER(dtype_to_ctype(dtype))), shape=shape)

    # Special case for float16
    if dtype == np.float16:
        array.dtype = np.float16

    return array

def cudaDrawText(image, text, position=(10, 10), color=None):
    """
    Draw text on a CUDA image using jetson_utils.cudaFont.
    Automatically creates a font object if not already defined.
    """
    _font = cudaFont()  
    color = _font.White

    try:
        _font.OverlayText(image, text, x=position[0], y=position[1], color=color)
    except Exception as e:
        print(f"[cudaDrawText] Failed to draw text: {e}")
