import numpy as np
import pygame


def interactive_visualisation_pygame(logic, n_rows, n_cols, tile_size=512, font_size=32):
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, font_size)

    initial_images, initial_titles = logic.reset()
    n_imgs = len(initial_images)
    assert n_imgs == n_rows * n_cols

    img_height, img_width = initial_images[0].shape[:2]
    screen_width = n_cols * tile_size
    screen_height = n_rows * (tile_size + font_size + 5)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Interactive Visualisation")

    def draw_images(images, titles):
        screen.fill((0, 0, 0))
        for idx, img in enumerate(images):
            row = idx // n_cols
            col = idx % n_cols
            x = col * tile_size
            y = row * (tile_size + font_size + 5)

            # Draw image
            # image is (H, W, C) -> needs to be (W, H, C) for surface.
            #  I checked that Image.fromarray(img).save("test.png") works as expected
            surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            surface = pygame.transform.scale(surface, (tile_size, tile_size))
            screen.blit(surface, (x, y + font_size + 5))

            # Draw title
            if titles[idx]:
                text_surface = font.render(titles[idx], True, (255, 255, 255))
                screen.blit(text_surface, (x, y))

        pygame.display.flip()

    def get_frame():
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))  # Convert from (width, height, channel) to (height, width, channel)
        return frame

    draw_images(initial_images, initial_titles)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                new_images, new_titles = logic.step(key_name)
                draw_images(new_images, new_titles)

    pygame.quit()


class VisualisationLogicWrapper:
    def __init__(self, env, seed, changing_seed=True):
        self.env = env
        self.seed = seed

        self._should_reset = False
        self._changing_seed = changing_seed

        self.key_map = {
            "left": 2,
            "right": 3,
            "up": 0,
            "down": 1,
        }

    def reset(self):
        self._should_reset = False
        if self._changing_seed:
            self.seed += 1

        obs, _ = self.env.reset(seed=self.seed)
        return [obs], ["Environment"]

    def step(self, key):
        if self._should_reset:
            return self.reset()

        action = self.key_map[key]
        obs, _, terminated, truncated, _ = self.env.step(action)
        return [obs], ["Environment"]
