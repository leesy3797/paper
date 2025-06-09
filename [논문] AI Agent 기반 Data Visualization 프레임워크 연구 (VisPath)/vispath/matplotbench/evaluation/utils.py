import base64
import io
import matplotlib.pyplot as plt

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def base64_to_image(base64_string, output_path):
    with open(output_path, "wb") as img_file:
        img_file.write(base64.b64decode(base64_string))

def code_to_image(code, img_save_path):
    import matplotlib.pyplot as plt
    exec_globals = {"plt": plt, "io": io}
    exec_locals = {}
    print('Start Executing Code and Save Final Image')
    try:
        code_n = code.replace("plt.show()", f"plt.savefig('{img_save_path}')\nplt.close('all')")
        exec(code_n, exec_globals, exec_locals)
        message = "Save Image Successfully!"
        print(message)
        return True
    except Exception as e:
        message = f"Error during Save : {str(e)}"
        print(message)
        return False