import Augmentor

p = Augmentor.Pipeline("./dataset/real")

p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)

p.flip_left_right(probability = 0.2)

p.flip_top_bottom(probability = 0.2)

p.rotate90(probability = 0.2)

p.sample(500)