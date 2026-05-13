import csv
from python_scripts.create_specifications import vnnlib_template_2
import random
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_properties.py <random_seed>")
        sys.exit(1)
        
    try:
        seed = int(sys.argv[1])
    except ValueError:
        print("Error: The random seed must be a numeric integer.")
        sys.exit(1)

    random.seed(seed)

    # create VNN-LIB 2.0 files given the following:
    VNN_COMP_TIMEOUT = 100  # per-instance verification timeout
    ONNX_MODEL_PATH = "onnx/acc-2000000-64-64-64-64-retrain-100000-200000-0.9.onnx"
    num_instances = 1

    boxes = []
    for i in range(5):
        for j in range(10):
            min_rPos = 0.0 + i * 20
            max_rPos = 0.0 + (i + 1) * 20
            min_rVel = -200.0 + j * 40
            max_rVel = -200.0 + (j + 1) * 40
            boxes.append((min_rPos, max_rPos, min_rVel, max_rVel))
            
    selected_boxes = random.sample(boxes, num_instances)

    instance_data = []
    for i, box in enumerate(selected_boxes, 1):
        min_rPos, max_rPos, min_rVel, max_rVel = box

        lines = vnnlib_template_2(min_rPos, max_rPos, min_rVel, max_rVel)

        vnnlib_filename = f"vnnlib/instance_{i}.vnnlib"
        with open(vnnlib_filename, "w") as f:
            f.writelines(line + "\n" for line in lines)

        instance = [ONNX_MODEL_PATH, vnnlib_filename, VNN_COMP_TIMEOUT]
        instance_data.append(instance)

    # save the ONNX/VNN-LIB instance pairs in the required CSV
    with open(f"instances.csv", 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(instance_data)


if __name__ == "__main__":
    main()