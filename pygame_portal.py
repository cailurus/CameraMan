import pygame
import pygame.camera

pygame.init()

gameDisplay = pygame.display.set_mode((1280,720))

pygame.camera.init()
print (pygame.camera.list_cameras())
cam = pygame.camera.Camera(0)
cam.start()
while True:
   img = cam.get_image()
   gameDisplay.blit(img,(0,0))
   pygame.display.update()
   for event in pygame.event.get() :
      if event.type == pygame.QUIT :
         cam.stop()
         pygame.quit()
         exit()