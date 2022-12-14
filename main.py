import math
import random

import cvzone
import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector

# change the parameter according to your pc
cap = cv2.VideoCapture(1)
# weight
cap.set(3, 1280)
# height
cap.set(4, 720)

# create hand detector object
# confidence is 80% and only can detect one hand
detector = HandDetector(detectionCon=0.8, maxHands=1)


# create class for snake game

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # list of whole points of a snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # current length of snake
        self.allowedLength = 150  # total allowed length of snake
        self.previousHead = 0, 0  # previous head position

        # the food
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        # dimension of the food, to make sure the snake ate the food
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()  # randomize the food location

        self.score = 0
        self.gameOver = False

    # generate random food location
    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    # display all the points and save it into correct list
    # update
    def update(self, imgMain, currentHead):

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550], scale=7, thickness=5, offset=20)
        else:
            # p is past and c is current

            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            # calculate distance between previous and current head
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            # increase the currentLength
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length Reduction to make sure the snake is below the allowed length
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    # drop the length and points until it is less than allowedLength
                    if self.currentLength < self.allowedLength:
                        break

            # check whether the snake ate the food
            rx, ry = self.foodPoint
            # if snake ate the food, generate a new random location
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                #print(self.score)

            # Draw the snake if there is any points
            if self.points:
                for i, point in enumerate(self.points):
                    # not the first point, because first point dont have the current head
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)

                # circle the last finger
                cv2.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

            # Draw the food
            rx, ry = self.foodPoint
            # to make sure the food image is overlayed in center.

            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

            cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80], scale=3, thickness=3, offset=20)
            # Check for Collision, except for second last point because not necessary
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            # print(minDist)

            if -1 <= minDist <= 1:
                print("Hit")
                self.gameOver = True
                self.points = []  # list of whole points of a snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # current length of snake
                self.allowedLength = 150  # total allowed length of snake
                self.previousHead = 0, 0  # previous head position
                self.randomFoodLocation()
        return imgMain


game = SnakeGameClass("Donut.png")

while (True):
    success, img = cap.read()
    # flip image in axis 1, so that it is easier to play the game(move the hand)
    img = cv2.flip(img, 1)
    # return total number of hands and list of landmarks
    hands, img = detector.findHands(img, flipType=False)

    # if there is a hand
    if hands:
        # get the first hands landmarks
        lmList = hands[0]['lmList']
        # index 8 is what we want (tip of the index finger)
        # get the x and y coordinates of the tip of the index finger
        pointIndex = lmList[8][0:2]
        # label the tip of the index finger as a circle
        cv2.circle(img, pointIndex, 20, (200, 0, 200), cv2.FILLED)
        img = game.update(img, pointIndex)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
        game.score = 0

    if key == ord('q'):
        break
