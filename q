[1mdiff --git a/stable_diffusion/webapp.py b/stable_diffusion/webapp.py[m
[1mindex bcc1602..c717e39 100644[m
[1m--- a/stable_diffusion/webapp.py[m
[1m+++ b/stable_diffusion/webapp.py[m
[36m@@ -33,7 +33,6 @@[m [mclass TextToImageBuilder():[m
         factory : Factory,[m
         max_num_images:int=9,[m
         images_per_row:int=3,[m
[31m-        device="cuda",[m
         ):[m
         """ Alternative constructor using a Factory[m
         """[m
[36m@@ -66,7 +65,7 @@[m [mclass TextToImageBuilder():[m
         )[m
         print(nsfw_content_detected)[m
 [m
[31m-        images_out = [auto_glue_image_grid(images)] + images + [None] * (self.max_num_images - len(images))[m
[32m+[m[32m        images_out = images + [None] * (self.max_num_images - len(images))[m
 [m
         return images_out[m
 [m
[36m@@ -92,9 +91,7 @@[m [mclass TextToImageBuilder():[m
 [m
         greet_btn = gr.Button("Generate !")[m
 [m
[31m-        out_glued_image = gr.Image(label=f"Generated images")[m
[31m-[m
[31m-        with gr.Accordion("Generated individual images"):[m
[32m+[m[32m        with gr.Accordion("Generated images"):[m
             i = 0[m
             output_images = [][m
             while i < self.max_num_images:[m
[36m@@ -104,7 +101,7 @@[m [mclass TextToImageBuilder():[m
                         out_img = gr.Image(label=f"Generated {i+1}")[m
                         output_images.append(out_img)[m
                         i += 1[m
[31m-        outputs = [out_glued_image] + output_images[m
[32m+[m[32m        outputs = output_images[m
         greet_btn.click(fn=self, inputs=inputs, outputs=outputs)[m
 [m
 [m
[36m@@ -128,7 +125,6 @@[m [mclass ImageToImageBuilder():[m
         factory : Factory,[m
         max_num_images:int=9,[m
         images_per_row:int=3,[m
[31m-        device="cuda",[m
         ):[m
         """ Alternative constructor using a Factory[m
         """[m
[36m@@ -161,7 +157,7 @@[m [mclass ImageToImageBuilder():[m
         )[m
         print(nsfw_content_detected)[m
 [m
[31m-        images_out = [auto_glue_image_grid(images)] + images + [None] * (self.max_num_images - len(images))[m
[32m+[m[32m        images_out = images + [None] * (self.max_num_images - len(images))[m
 [m
         return images_out[m
 [m
[36m@@ -186,8 +182,6 @@[m [mclass ImageToImageBuilder():[m
 [m
         greet_btn = gr.Button("Generate !")[m
 [m
[31m-        out_glued_image = gr.Image(label=f"Generated images")[m
[31m-[m
         with gr.Accordion("Generated individual images"):[m
             i = 0[m
             output_images = [][m
[36m@@ -198,7 +192,7 @@[m [mclass ImageToImageBuilder():[m
                         out_img = gr.Image(label=f"Generated {i+1}")[m
                         output_images.append(out_img)[m
                         i += 1[m
[31m-        outputs = [out_glued_image] + output_images[m
[32m+[m[32m        outputs = output_images[m
         greet_btn.click(fn=self, inputs=inputs, outputs=outputs)[m
 [m
 [m
