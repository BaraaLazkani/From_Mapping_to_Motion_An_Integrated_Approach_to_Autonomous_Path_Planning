import pygame, os, math, time, random, copy
from pygame.locals import *
from PIL import Image

pygame.init()

BARRIERRADIUS = 0.1
ROBOTRADIUS = 0.10
W = 2 * ROBOTRADIUS 
SAFEDIST = ROBOTRADIUS     

MAXVELOCITY = 0.3    
MAXACCELERATION = 0.5 

BARRIERVELOCITYRANGE = 0.1

PLAYFIELDCORNERS = (-4.0, -3.0, 4.0, 3.0)

x = PLAYFIELDCORNERS[0] - 0.5
y = 0.0
theta = 0.0

locationhistory = []

vL = 0.00
vR = 0.00

dt = 0.1
STEPSAHEADTOPLAN = 10
TAU = dt * STEPSAHEADTOPLAN

barriers = []
static_barriers = [
    (-2.0, 1.0),  
    (2.0, 1.0),
    (0.0, 2.0)
]
for i in range(35):
    (bx, by, vx, vy) = (random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2]), 
                        random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]), 
                        random.gauss(0.0, BARRIERVELOCITYRANGE), 
                        random.gauss(0.0, BARRIERVELOCITYRANGE))
    barrier = [bx, by, vx, vy]
    state= False
    for x,y in static_barriers:
    	if (bx<=x+7*BARRIERRADIUS and bx>=x-7*BARRIERRADIUS) or (by<=y+7*BARRIERRADIUS and by>=y-7*BARRIERRADIUS):
            state=True
    if state ==False:		
  	  barriers.append(barrier)

targetindex = random.randint(0, len(barriers))




WIDTH = 1500
HEIGHT = 1000
size = [WIDTH, HEIGHT]
black = (0,0,0)
lightblue = (0,120,255)
darkblue = (0,40,160)
green = (0, 255, 0)
red = (255,0,0)
white = (255,255,255)
blue = (0,0,255)
grey = (70,70,70)
k = 160
u0 = WIDTH / 2
v0 = HEIGHT / 2

screen = pygame.display.set_mode(size)
pygame.mouse.set_visible(0)

pathstodraw = []

# GIF recording settings
RECORD_GIF = True  # Set to False to disable recording
GIF_DURATION = 15  # seconds
GIF_FILENAME = "simulation.gif"
FRAME_SKIP = 5  # Only capture every Nth frame to reduce file size (higher = smaller file)
frames = []
recording = False
start_time = None
frame_count = 0

def predictPosition(vL, vR, x, y, theta, deltat):
    if (round(vL, 3) == round(vR, 3)):
        xnew = x + vL * deltat * math.cos(theta)
        ynew = y + vL * deltat * math.sin(theta)
        thetanew = theta
        path = (0, vL * deltat)   
    elif (round(vL, 3) == -round(vR, 3)):
        xnew = x
        ynew = y
        thetanew = theta + ((vR - vL) * deltat / W)
        path = (1, 0)  
    else:
        R = W / 2.0 * (vR + vL) / (vR - vL)
        deltatheta = (vR - vL) * deltat / W
        xnew = x + R * (math.sin(deltatheta + theta) - math.sin(theta))
        ynew = y - R * (math.cos(deltatheta + theta) - math.cos(theta))
        thetanew = theta + deltatheta

        cx = x - R * math.sin(theta)
        cy = y + R * math.cos(theta)
        Rabs = abs(R)
        ((tlx, tly), (Rx, Ry)) = ((int(u0 + k * (cx - Rabs)), int(v0 - k * (cy + Rabs))), 
                                  (int(k * (2 * Rabs)), int(k * (2 * Rabs))))
        if (R > 0):
            start_angle = theta - math.pi/2.0
        else:
            start_angle = theta + math.pi/2.0
        stop_angle = start_angle + deltatheta
        path = (2, ((tlx, tly), (Rx, Ry)), start_angle, stop_angle)

    return (xnew, ynew, thetanew, path)

def calculateClosestObstacleDistance(x, y):
    closestdist = 100000.0
    for (i, barrier) in enumerate(barriers):
        if (i != targetindex):
            dx = barrier[0] - x
            dy = barrier[1] - y
            d = math.sqrt(dx**2 + dy**2)
            dist = d - BARRIERRADIUS - ROBOTRADIUS
            if (dist < closestdist):
                closestdist = dist

    
    for static in static_barriers:
        dx = static[0] - x
        dy = static[1] - y
        d = math.sqrt(dx**2 + dy**2)
        dist = d - 7*BARRIERRADIUS - ROBOTRADIUS
        if (dist < closestdist):
            closestdist = dist

    return closestdist

def drawBarriers(barriers, static_barriers):
    for (i, barrier) in enumerate(barriers):
        if (i == targetindex):
            bcol = lightblue
        else:
            bcol = (139, 0, 0)
        pygame.draw.circle(screen, bcol, (int(u0 + k * barrier[0]), int(v0 - k * barrier[1])), int(k* BARRIERRADIUS), 0)
    
    
    for static in static_barriers:
        pygame.draw.circle(screen, grey, (int(u0 + k * static[0]), int(v0 - k * static[1])), int(1000 * BARRIERRADIUS), 0)

def moveBarriers(dt):
    
    for (i, barrier) in enumerate(barriers):
        barriers[i][0] += barriers[i][2] * dt
        barriers[i][1] += barriers[i][3] * dt
    
        if (barriers[i][0] < PLAYFIELDCORNERS[0] or barriers[i][0] > PLAYFIELDCORNERS[2]):
            barriers[i][2] = -barriers[i][2]
        if (barriers[i][1] < PLAYFIELDCORNERS[1] or barriers[i][1] > PLAYFIELDCORNERS[3]):
            barriers[i][3] = -barriers[i][3]

        
        for static in static_barriers:
            dx = barrier[0] - static[0]
            dy = barrier[1] - static[1]
            d = math.sqrt(dx**2 + dy**2)
            if d < 7* BARRIERRADIUS:
                barriers[i][2] = -barriers[i][2]
                barriers[i][3] = -barriers[i][3]

			

while True:
    Eventlist = pygame.event.get()

    # Start recording after first frame
    if RECORD_GIF and not recording:
        recording = True
        start_time = time.time()
        print("Started recording GIF...")

    locationhistory.append((x, y))
    bestBenefit = -100000
    FORWARDWEIGHT = 12
    OBSTACLEWEIGHT = 6666

    barrierscopy = copy.deepcopy(barriers)

    for i in range(STEPSAHEADTOPLAN):
        moveBarriers(dt)

    vLpossiblearray = (vL - MAXACCELERATION * dt, vL, vL + MAXACCELERATION * dt)
    vRpossiblearray = (vR - MAXACCELERATION * dt, vR, vR + MAXACCELERATION * dt)
    pathstodraw = [] 
    newpositionstodraw = [] 
    for vLpossible in vLpossiblearray:
        for vRpossible in vRpossiblearray:
            if (vLpossible <= MAXVELOCITY and vRpossible <= MAXVELOCITY and 
                vLpossible >= -MAXVELOCITY and vRpossible >= -MAXVELOCITY):
                (xpredict, ypredict, thetapredict, path) = predictPosition(vLpossible, vRpossible, x, y, theta, TAU)
                pathstodraw.append(path)
                newpositionstodraw.append((xpredict, ypredict))
                distanceToObstacle = calculateClosestObstacleDistance(xpredict, ypredict)
                previousTargetDistance = math.sqrt((x - barriers[targetindex][0])**2 + (y - barriers[targetindex][1])**2)
                newTargetDistance = math.sqrt((xpredict - barriers[targetindex][0])**2 + (ypredict - barriers[targetindex][1])**2)
                distanceForward = previousTargetDistance - newTargetDistance
                distanceBenefit = FORWARDWEIGHT * distanceForward
                if (distanceToObstacle < SAFEDIST):
                    obstacleCost = OBSTACLEWEIGHT * (SAFEDIST - distanceToObstacle)
                else:
                    obstacleCost = 0.0
                benefit = distanceBenefit - obstacleCost
                if (benefit > bestBenefit):
                    vLchosen = vLpossible
                    vRchosen = vRpossible
                    bestBenefit = benefit

    vL = vLchosen
    vR = vRchosen
    barriers = copy.deepcopy(barrierscopy)

    screen.fill((20,20,20))
    for loc in locationhistory:
        pygame.draw.circle(screen, (255,255,250), (int(u0 + k * loc[0]), int(v0 - k * loc[1])),1,0)
    drawBarriers(barriers, static_barriers)

    u = u0 + k * x
    v = v0 - k * y
    pygame.draw.circle(screen, white, (int(u), int(v)), int(k * ROBOTRADIUS), 0)
    wlx = x - (W/2.0) * math.sin(theta)
    wly = y + (W/2.0) * math.cos(theta)
    ulx = u0 + k * wlx
    vlx = v0 - k * wly
    WHEELBLOB = 0.04
    pygame.draw.circle(screen, blue, (int(ulx), int(vlx)), int(k * WHEELBLOB))
    wrx = x + (W/2.0) * math.sin(theta)
    wry = y - (W/2.0) * math.cos(theta)
    urx = u0 + k * wrx
    vrx = v0 - k * wry
    pygame.draw.circle(screen, blue, (int(urx), int(vrx)), int(k * WHEELBLOB))

    for path in pathstodraw:
        if path[0] == 0:    
            straightpath = path[1]
            linestart = (u0 + k * x, v0 - k * y)
            lineend = (u0 + k * (x + straightpath * math.cos(theta)), 
                       v0 - k * (y + straightpath * math.sin(theta)))
            pygame.draw.line(screen, green, linestart, lineend, 0)
        if path[0] == 2:    
            if (path[3] > path[2]):
                startangle = path[2]
                stopangle = path[3]
            else:
                startangle = path[3]
                stopangle = path[2]
            if (startangle < 0):
                startangle += 2*math.pi
                stopangle += 2*math.pi
            if (path[1][1][0] > 0 and path[1][0][0] > 0 and path[1][1][1] > 1):
                pygame.draw.arc(screen, green, path[1], startangle, stopangle, 1)

    pygame.display.flip()

    # Capture frame for GIF
    if RECORD_GIF and recording:
        elapsed_time = time.time() - start_time
        if elapsed_time <= GIF_DURATION:
            # Only capture every FRAME_SKIP frames to reduce size
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                # Capture and scale down the screen
                frame_surface = pygame.Surface(screen.get_size())
                frame_surface.blit(screen, (0, 0))

                # Scale down to 50% to reduce memory/file size
                scaled_surface = pygame.transform.scale(frame_surface, (WIDTH // 2, HEIGHT // 2))

                frame_string = pygame.image.tostring(scaled_surface, 'RGB')
                frame_image = Image.frombytes('RGB', (WIDTH // 2, HEIGHT // 2), frame_string)
                frames.append(frame_image)

                # Print progress
                if len(frames) % 20 == 0:
                    print(f"Recording: {elapsed_time:.1f}/{GIF_DURATION} seconds ({len(frames)} frames)")
        else:
            # Save GIF and stop recording
            print(f"Saving GIF with {len(frames)} frames... Please wait.")
            try:
                frames[0].save(
                    GIF_FILENAME,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(dt * 1000 / 50 * FRAME_SKIP),  # Adjust for frame skip
                    loop=0,
                    optimize=False  # Faster saving
                )
                print(f"GIF saved successfully as {GIF_FILENAME}")
            except Exception as e:
                print(f"Error saving GIF: {e}")
            finally:
                recording = False
                RECORD_GIF = False
                frames = []  # Free memory

    (x, y, theta, tmppath) = predictPosition(vL, vR, x, y, theta, dt)
    moveBarriers(dt)

    disttotarget = math.sqrt((x - barriers[targetindex][0])**2 + (y - barriers[targetindex][1])**2)
    if (disttotarget < (BARRIERRADIUS + ROBOTRADIUS)):
        for i in range(0):
            (bx, by) = (random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2]), 
                        random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]))
            (bx, by, vx, vy) = (random.uniform(PLAYFIELDCORNERS[0], PLAYFIELDCORNERS[2]), 
                                random.uniform(PLAYFIELDCORNERS[1], PLAYFIELDCORNERS[3]), 
                                random.uniform(-BARRIERVELOCITYRANGE, BARRIERVELOCITYRANGE), 
                                random.uniform(-BARRIERVELOCITYRANGE, BARRIERVELOCITYRANGE))
            barrier = [bx, by, vx, vy]
            barriers.append(barrier)
        targetindex = random.randint(0, len(barriers)-1)
        locationhistory = []
    time.sleep(dt / 50)