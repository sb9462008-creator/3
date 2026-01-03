import pygame
import random
import sys

pygame.init()

# Өнгө
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
PURPLE = (200, 0, 255)
GRAY = (128, 128, 128)

# Тохиргоо
CELL_SIZE = 30
GRID_WIDTH = 25
GRID_HEIGHT = 20
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT + 100
BASE_FPS = 8

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('🐍 Snake Game - Advanced')
clock = pygame.time.Clock()

font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)


class Snake:
    def __init__(self):
        self.body = [[GRID_WIDTH // 2, GRID_HEIGHT // 2]]
        self.direction = [1, 0]
        self.grow = False
        self.alive = True
    
    def move(self):
        if not self.alive:
            return
        
        new_head = [
            self.body[0][0] + self.direction[0],
            self.body[0][1] + self.direction[1]
        ]
        
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.alive = False
            return
        
        if new_head in self.body:
            self.alive = False
            return
        
        self.body.insert(0, new_head)
        
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
    
    def change_direction(self, new_direction):
        if (new_direction[0] != -self.direction[0] or 
            new_direction[1] != -self.direction[1]):
            self.direction = new_direction
    
    def eat(self):
        self.grow = True
    
    def draw(self, surface):
        for i, segment in enumerate(self.body):
            x = segment[0] * CELL_SIZE
            y = segment[1] * CELL_SIZE + 100
            
            if i == 0:
                pygame.draw.rect(surface, GREEN, 
                               (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4), 
                               border_radius=8)
                
                eye_size = 6
                if self.direction == [1, 0]:
                    pygame.draw.circle(surface, WHITE, 
                                     (x + CELL_SIZE - 10, y + 8), eye_size)
                    pygame.draw.circle(surface, WHITE, 
                                     (x + CELL_SIZE - 10, y + CELL_SIZE - 8), eye_size)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + CELL_SIZE - 10, y + 8), 3)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + CELL_SIZE - 10, y + CELL_SIZE - 8), 3)
                elif self.direction == [-1, 0]:
                    pygame.draw.circle(surface, WHITE, 
                                     (x + 10, y + 8), eye_size)
                    pygame.draw.circle(surface, WHITE, 
                                     (x + 10, y + CELL_SIZE - 8), eye_size)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + 10, y + 8), 3)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + 10, y + CELL_SIZE - 8), 3)
                elif self.direction == [0, -1]:
                    pygame.draw.circle(surface, WHITE, 
                                     (x + 8, y + 10), eye_size)
                    pygame.draw.circle(surface, WHITE, 
                                     (x + CELL_SIZE - 8, y + 10), eye_size)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + 8, y + 10), 3)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + CELL_SIZE - 8, y + 10), 3)
                else:
                    pygame.draw.circle(surface, WHITE, 
                                     (x + 8, y + CELL_SIZE - 10), eye_size)
                    pygame.draw.circle(surface, WHITE, 
                                     (x + CELL_SIZE - 8, y + CELL_SIZE - 10), eye_size)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + 8, y + CELL_SIZE - 10), 3)
                    pygame.draw.circle(surface, BLACK, 
                                     (x + CELL_SIZE - 8, y + CELL_SIZE - 10), 3)
            else:
                pygame.draw.rect(surface, DARK_GREEN, 
                               (x + 4, y + 4, CELL_SIZE - 8, CELL_SIZE - 8),
                               border_radius=5)


class Food:
    def __init__(self, snake_body):
        self.position = self.generate_position(snake_body)
        self.type = random.choice(['normal', 'bonus'])
    
    def generate_position(self, snake_body):
        while True:
            pos = [random.randint(0, GRID_WIDTH - 1),
                   random.randint(0, GRID_HEIGHT - 1)]
            if pos not in snake_body:
                return pos
    
    def draw(self, surface):
        x = self.position[0] * CELL_SIZE + CELL_SIZE // 2
        y = self.position[1] * CELL_SIZE + 100 + CELL_SIZE // 2
        
        if self.type == 'normal':
            # Энгийн хоол (улаан)
            pygame.draw.circle(surface, RED, (x, y), CELL_SIZE // 2 - 4)
            pygame.draw.rect(surface, (139, 69, 19), 
                            (x - 2, y - CELL_SIZE // 2 + 2, 4, 8))
        else:
            # Bonus хоол (алтан)
            pygame.draw.circle(surface, YELLOW, (x, y), CELL_SIZE // 2 - 4)
            pygame.draw.circle(surface, (255, 215, 0), (x, y), CELL_SIZE // 2 - 8)
            # Од
            for angle in range(0, 360, 72):
                import math
                x1 = x + math.cos(math.radians(angle)) * (CELL_SIZE // 4)
                y1 = y + math.sin(math.radians(angle)) * (CELL_SIZE // 4)
                pygame.draw.circle(surface, WHITE, (int(x1), int(y1)), 3)


class Obstacle:
    """Саад (Түвшин дээр үүснэ)"""
    def __init__(self, snake_body, food_pos):
        self.position = self.generate_position(snake_body, food_pos)
    
    def generate_position(self, snake_body, food_pos):
        while True:
            pos = [random.randint(0, GRID_WIDTH - 1),
                   random.randint(0, GRID_HEIGHT - 1)]
            if pos not in snake_body and pos != food_pos:
                return pos
    
    def draw(self, surface):
        x = self.position[0] * CELL_SIZE
        y = self.position[1] * CELL_SIZE + 100
        pygame.draw.rect(surface, GRAY, 
                        (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))
        # X тэмдэг
        pygame.draw.line(surface, RED, 
                        (x + 5, y + 5), 
                        (x + CELL_SIZE - 5, y + CELL_SIZE - 5), 3)
        pygame.draw.line(surface, RED, 
                        (x + CELL_SIZE - 5, y + 5), 
                        (x + 5, y + CELL_SIZE - 5), 3)


def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRAY, (x, 100), (x, SCREEN_HEIGHT), 1)
    for y in range(100, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRAY, (0, y), (SCREEN_WIDTH, y), 1)


def draw_header(surface, score, high_score, level):
    pygame.draw.rect(surface, DARK_GREEN, (0, 0, SCREEN_WIDTH, 100))
    
    title = font_large.render('🐍 SNAKE', True, YELLOW)
    surface.blit(title, (20, 10))
    
    score_text = font_medium.render(f'Оноо: {score}', True, WHITE)
    surface.blit(score_text, (20, 55))
    
    level_text = font_medium.render(f'Түвшин: {level}', True, WHITE)
    surface.blit(level_text, (SCREEN_WIDTH // 2 - level_text.get_width() // 2, 55))
    
    high_text = font_medium.render(f'Дээд: {high_score}', True, WHITE)
    surface.blit(high_text, (SCREEN_WIDTH - high_text.get_width() - 20, 55))


def draw_game_over(surface, score, high_score, level):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))
    
    game_over_text = font_large.render('ТОГЛООМ ДУУСЛАА!', True, RED)
    surface.blit(game_over_text, 
                (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                 SCREEN_HEIGHT // 2 - 120))
    
    score_text = font_medium.render(f'Оноо: {score}', True, WHITE)
    surface.blit(score_text, 
                (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 
                 SCREEN_HEIGHT // 2 - 60))
    
    level_text = font_medium.render(f'Хүрсэн түвшин: {level}', True, WHITE)
    surface.blit(level_text, 
                (SCREEN_WIDTH // 2 - level_text.get_width() // 2, 
                 SCREEN_HEIGHT // 2 - 20))
    
    if score == high_score and score > 0:
        new_record = font_medium.render('🏆 ШИНЭ РЕКОРД! 🏆', True, YELLOW)
        surface.blit(new_record, 
                    (SCREEN_WIDTH // 2 - new_record.get_width() // 2, 
                     SCREEN_HEIGHT // 2 + 20))
    
    restart_text = font_small.render('SPACE - Дахин эхлүүлэх', True, WHITE)
    surface.blit(restart_text, 
                (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                 SCREEN_HEIGHT // 2 + 70))
    
    quit_text = font_small.render('ESC - Гарах', True, WHITE)
    surface.blit(quit_text, 
                (SCREEN_WIDTH // 2 - quit_text.get_width() // 2, 
                 SCREEN_HEIGHT // 2 + 100))


def main():
    snake = Snake()
    food = Food(snake.body)
    obstacles = []
    score = 0
    high_score = 0
    level = 1
    game_over = False
    fps = BASE_FPS
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if not game_over:
                    if event.key in [pygame.K_UP, pygame.K_w]:
                        snake.change_direction([0, -1])
                    elif event.key in [pygame.K_DOWN, pygame.K_s]:
                        snake.change_direction([0, 1])
                    elif event.key in [pygame.K_LEFT, pygame.K_a]:
                        snake.change_direction([-1, 0])
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        snake.change_direction([1, 0])
                else:
                    if event.key == pygame.K_SPACE:
                        snake = Snake()
                        food = Food(snake.body)
                        obstacles = []
                        score = 0
                        level = 1
                        fps = BASE_FPS
                        game_over = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
        
        if not game_over:
            snake.move()
            
            if not snake.alive:
                game_over = True
                if score > high_score:
                    high_score = score
            
            # Саадтай мөргөлдөх
            for obstacle in obstacles:
                if snake.body[0] == obstacle.position:
                    snake.alive = False
                    game_over = True
            
            # Хоол идэх
            if snake.body[0] == food.position:
                snake.eat()
                
                if food.type == 'normal':
                    score += 10
                else:  # bonus
                    score += 50
                
                # Түвшин өсгөх
                if score // 100 + 1 > level:
                    level = score // 100 + 1
                    fps = BASE_FPS + level  # Хурд нэмэгдэх
                    
                    # Саад нэмэх
                    if level > 2:
                        obstacles.append(Obstacle(snake.body, food.position))
                
                food = Food(snake.body)
        
        # Зурах
        screen.fill(BLACK)
        draw_grid(screen)
        draw_header(screen, score, high_score, level)
        
        for obstacle in obstacles:
            obstacle.draw(screen)
        
        food.draw(screen)
        snake.draw(screen)
        
        if game_over:
            draw_game_over(screen, score, high_score, level)
        
        pygame.display.flip()
        clock.tick(fps)
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()