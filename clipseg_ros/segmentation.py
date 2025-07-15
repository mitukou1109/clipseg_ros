import ast
import copy
import threading
import time

import cv2
import cv_bridge
import numpy as np
import numpy.typing as npt
import rclpy
import rclpy.callback_groups
import rclpy.node
import sensor_msgs.msg
import std_msgs.msg
import torch
import transformers
import transformers.models.clipseg.modeling_clipseg
import transformers.utils.logging


class Segmentation(rclpy.node.Node):
    class Result:
        def __init__(
            self,
            labels: npt.NDArray[np.uint8],
            source_image: npt.NDArray[np.uint8],
        ):
            self.labels = labels
            self.source_image = source_image

    def __init__(self):
        super().__init__("segmentation")

        use_gpu = (
            self.declare_parameter("use_gpu", True).get_parameter_value().bool_value
        )
        pretrained_model_name_or_path = (
            self.declare_parameter(
                "pretrained_model_name_or_path", "CIDAS/clipseg-rd64-refined"
            )
            .get_parameter_value()
            .string_value
        )
        self.class_prompts = ast.literal_eval(
            self.declare_parameter("class_prompts", "[]")
            .get_parameter_value()
            .string_value
        )
        self.class_colors = ast.literal_eval(
            self.declare_parameter("class_colors", "[]")
            .get_parameter_value()
            .string_value
        )
        self.score_threshold = (
            self.declare_parameter("score_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        self.enable_padding = (
            self.declare_parameter("enable_padding", True)
            .get_parameter_value()
            .bool_value
        )
        self.use_compressed_image = (
            self.declare_parameter("use_compressed_image", True)
            .get_parameter_value()
            .bool_value
        )
        result_visualization_rate = (
            self.declare_parameter("result_visualization_rate", 10.0)
            .get_parameter_value()
            .double_value
        )

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.clipseg_processor = transformers.CLIPSegProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=True
        )
        self.clipseg_model = transformers.CLIPSegForImageSegmentation.from_pretrained(
            pretrained_model_name_or_path
        ).to(self.device)
        transformers.utils.logging.set_verbosity_error()

        self.cv_bridge = cv_bridge.CvBridge()

        self.result_lock = threading.Lock()
        self.result: Segmentation.Result = None

        self.label_image_pub = self.create_publisher(
            sensor_msgs.msg.Image,
            "~/label_image",
            1,
        )

        if self.use_compressed_image:
            self.image_raw_sub = self.create_subscription(
                sensor_msgs.msg.CompressedImage,
                "image_raw/compressed",
                self.image_raw_compressed_callback,
                1,
            )
        else:
            self.image_raw_sub = self.create_subscription(
                sensor_msgs.msg.Image,
                "image_raw",
                self.image_raw_callback,
                1,
            )

        self.visualize_result_timer = self.create_timer(
            1 / result_visualization_rate,
            self.visualize_result_callback,
            rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
        )

    def image_raw_compressed_callback(self, msg: sensor_msgs.msg.CompressedImage):
        if not msg.data:
            self.get_logger().warn("Image data is empty")
            return
        source_image = self.cv_bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="rgb8"
        )
        self.run_segmentation(source_image, msg.header)

    def image_raw_callback(self, msg: sensor_msgs.msg.Image):
        if not msg.data:
            self.get_logger().warn("Image data is empty")
            return
        source_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.run_segmentation(source_image, msg.header)

    def visualize_result_callback(self):
        try:
            self.class_colors = ast.literal_eval(
                self.get_parameter("class_colors").get_parameter_value().string_value
            )
        except Exception as e:
            self.get_logger().error(f"Parameter class_colors has invalid format: {e}")
            return

        if len(self.class_colors) != len(self.class_prompts):
            self.get_logger().error(
                "Parameter class_colors must have the same length as class_prompts"
            )
            return

        if self.result is None:
            return

        with self.result_lock:
            result = copy.deepcopy(self.result)

        result_image = cv2.cvtColor(
            cv2.addWeighted(
                src1=np.array([[0, 0, 0]] + self.class_colors, dtype=np.uint8)[
                    result.labels
                ],
                src2=result.source_image,
                alpha=0.5,
                beta=1.0,
                gamma=0.0,
            ),
            cv2.COLOR_RGB2BGR,
        )

        cv2.imshow(self.get_name(), result_image)
        cv2.waitKey(1)

    def run_segmentation(
        self, source_image: npt.NDArray[np.uint8], header: std_msgs.msg.Header
    ):
        try:
            self.class_prompts = ast.literal_eval(
                self.get_parameter("class_prompts").get_parameter_value().string_value
            )
            self.score_threshold = (
                self.get_parameter("score_threshold").get_parameter_value().double_value
            )
        except Exception as e:
            self.get_logger().error(f"Parameter has invalid format: {e}")
            return

        start_time = time.time_ns()

        input_image = (
            self.pad_image_to_square(source_image)
            if self.enable_padding
            else source_image
        )

        inputs = self.clipseg_processor(
            text=self.class_prompts,
            images=[torch.from_numpy(input_image).to(self.device).to(torch.uint8)]
            * len(self.class_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs: (
                transformers.models.clipseg.modeling_clipseg.CLIPSegImageSegmentationOutput
            ) = self.clipseg_model(**inputs)
        scores = outputs.logits.unsqueeze(1)

        scores: torch.Tensor = torch.nn.functional.interpolate(
            scores,
            size=source_image.shape[:2],
            mode="bilinear",
            align_corners=True,
        )
        scores = scores.squeeze(1)

        scores_masked = torch.where(
            scores >= self.score_threshold,
            scores,
            torch.tensor(-torch.inf, device=self.device),
        )

        max_scores, max_indices = torch.max(scores_masked, dim=0)
        max_indices = max_indices.to(torch.uint8)

        labels = torch.zeros_like(max_indices, device=self.device)
        valid_mask = max_scores != -torch.inf
        labels[valid_mask] = max_indices[valid_mask] + 1

        labels = labels.cpu().numpy()

        self.get_logger().debug(
            f"Segmentation took {(time.time_ns() - start_time) / 1e6:.2f} ms"
        )

        label_image_msg = self.cv_bridge.cv2_to_imgmsg(
            labels,
            encoding="8UC1",
            header=header,
        )
        self.label_image_pub.publish(label_image_msg)

        with self.result_lock:
            self.result = Segmentation.Result(labels, source_image)

    def pad_image_to_square(
        self, source_image: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        height, width = source_image.shape[:2]
        target_size = max(height, width)

        pad_top = (target_size - height) // 2
        pad_bottom = target_size - height - pad_top
        pad_left = (target_size - width) // 2
        pad_right = target_size - width - pad_left

        padded_image = np.pad(
            source_image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

        return padded_image


def main(args=None):
    rclpy.init(args=args)

    segmentation = Segmentation()

    rclpy.spin(segmentation)

    segmentation.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
