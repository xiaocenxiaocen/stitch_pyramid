# file: stitch.py
# author: xiaocenxiaocen
# date: 2016.11.25
'''
Prototype program for stitching pyramid images into a large plane
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plan_optimizer(imageLevels):
	width = imageLevels[0][1]
	height = imageLevels[0][0]

	wMax = 0
	hMax = 0
	wList = []
	for img in imageLevels:
		wMax += img[1]
		hMax += img[0]
		wList.append(img[1])
	
	print imageLevels
	print wList
	wTable = get_table(wList)
	wTable = filter(lambda x : x >= width, wTable)
	
	print wTable
	minArea = min(width * hMax, height * wMax)
	optPlan = []
	
	for w in wTable:
		plan = make_stitch_plan_(imageLevels, height, w)
		area = get_plan_area(imageLevels, plan)
		print plan, area
		if area <= minArea:
			optPlan = plan
			minArea = area
	
	return optPlan, minArea

def get_table(wList):
	if len(wList) == 0: return []
	if len(wList) == 1: return wList
	if len(wList) > 1:
		wTable = get_table(wList[1:])
		wTableAppend = [wList[0]]
		for w in wTable:
			wTableAppend.append(w + wList[0])
		wTable = wTable + wTableAppend
		wTable = uniqify_sheets(wTable)
		return wTable
			
def uniqify_sheets(sheets):
	def idfun(x):
		return str(x)
	seen = {}
	result = []
	for sheet in sheets:
		marker = idfun(sheet)
		if marker in seen: continue
		seen[marker] = 1
		result.append(sheet)
	return result

def make_stitch_plan(imageLevels, MAX_COL, MAX_ROW):
	'''
	Bottom Left Fill Algorithm
	the filling process is not from lower sheet to upper sheet
	'''
	sheets = []
	sheets.append(np.array([0, 0, MAX_COL, MAX_ROW], dtype = np.integer))
	
	plan = []
	
	top = MAX_COL
	for image in imageLevels:
		hImage = image[0]
		wImage = image[1]		
		
		findSheet = False
		bottomImage = int(0)
		leftImage = int(0)
		sheetIdx = int(0)
		
		for idx, sheet in enumerate(sheets):
			bottomSheet = sheet[0]
			leftSheet = sheet[1]
			hSheet = sheet[2]
			wSheet = sheet[3]
			if hSheet >= hImage and wSheet >= wImage:
				findSheet = True
				bottomImage = bottomSheet
				leftImage = leftSheet
				sheetIdx = idx
				plan.append((bottomImage, leftImage))
				break
			
		if not findSheet:
			sheets.append(np.array([top, 0, hImage, MAX_ROW], dtype = np.integer))
			sheetIdx = len(sheets) - 1
			bottomImage = top
			leftImage = 0
			plan.append((top, 0))
			top += hImage
		
		topImage = bottomImage + hImage
		rightImage = leftImage + wImage
		
		idx = 0
		while(idx < len(sheets)):
			bottomSheet = sheets[idx][0]
			leftSheet = sheets[idx][1]
			topSheet = bottomSheet + sheets[idx][2]
			rightSheet = leftSheet + sheets[idx][3]
			if not (bottomImage >= topSheet or topImage <= bottomSheet or leftImage >= rightSheet or rightImage <= leftSheet):
				if leftImage > leftSheet:
					sheets.append(np.array([bottomSheet, leftSheet, sheets[idx][2], leftImage - leftSheet], dtype = np.integer))
				
				if bottomImage > bottomSheet:
					sheets.append(np.array([bottomSheet, leftSheet, bottomImage - bottomSheet, sheets[idx][3]], dtype = np.integer))
				
				if rightImage < rightSheet:
					sheets.append(np.array([bottomSheet, rightImage, sheets[idx][2], rightSheet - rightImage], dtype = np.integer))	
				
				if topImage < topSheet:
					sheets.append(np.array([topImage, leftSheet, topSheet - topImage, sheets[idx][3]], dtype = np.integer))
				
				sheets.pop(idx)
			#	idx += 1
			else:
				idx += 1
		
			sheets = uniqify_sheets(sheets)
		
	return plan

def make_stitch_plan_(imageLevels, MAX_COL, MAX_ROW):
	'''
	@param: imageLevels, list, size of pyramid images
	@param: MAX_COL, MAX_ROW, size of the first sheet appended into the sheets list, MAX_COL >= original image collum, MAX_ROW >= original image row
	@ret: plan, list, first sample in x-direction and y-direction of each image in the large plane
	'''
	
	'''
	Bottom Left Fill Algorithm
	Reference:
	Edmund Burke et. al., 2006, A New Bottom-Left-Fill Heuristic Algorithm for the Two-Dimensional Irregular Packing Problem. OPERATIONS RESEARCH, Vol. 54, No. 3, pp. 587-601.
	
	guthub: https://github.com/forresti/caffe/tree/2c2ec6413d947bd19edc20fdbc36e7dcef0dcb70/src/stitch_pyramid
	file: Patchwork.cpp, subroutine: int Patchwork::BLF(vector<pair<Rectangle, int> > & retangles)
	
	'''
	# sheet: two dimensional list, the filling process is from lower sheet to upper sheet
	sheets = []
	sheets.append(list())
	sheets[0].append(np.array([0, 0, MAX_COL, MAX_ROW], dtype = np.integer))
	
	plan = []
	
	top = MAX_COL
	for i, image in enumerate(imageLevels):
		hImage = image[0]
		wImage = image[1]		
		
		findGap = False
		bottomImage = int(0)
		leftImage = int(0)
		
		for idx, sheet in enumerate(sheets):
			for gap in sheet:
				bottomGap = gap[0]
				leftGap = gap[1]
				hGap = gap[2]
				wGap = gap[3]
				if hGap >= hImage and wGap >= wImage:
					findGap = True
					bottomImage = bottomGap
					leftImage = leftGap
					sheetIdx = idx
					plan.append((bottomImage, leftImage))
					break
			if findGap: break
			
		if not findGap:
			sheets.append(list())
			sheets[-1].append(np.array([top, 0, hImage, MAX_ROW], dtype = np.integer))
			sheetIdx = len(sheets) - 1
			bottomImage = top
			leftImage = 0
			plan.append((top, 0))
			top += hImage
		
		topImage = bottomImage + hImage
		rightImage = leftImage + wImage
		
		idx = 0
		while idx < len(sheets[sheetIdx]):
			bottomGap = sheets[sheetIdx][idx][0]
			leftGap = sheets[sheetIdx][idx][1]
			topGap = bottomGap + sheets[sheetIdx][idx][2]
			rightGap = leftGap + sheets[sheetIdx][idx][3]
			if not (bottomImage >= topGap or topImage <= bottomGap or leftImage >= rightGap or rightImage <= leftGap):
				if leftImage > leftGap:
					sheets[sheetIdx].append(np.array([bottomGap, leftGap, sheets[sheetIdx][idx][2], leftImage - leftGap], dtype = np.integer))
				
				if bottomImage > bottomGap:
					sheets[sheetIdx].append(np.array([bottomGap, leftGap, bottomImage - bottomGap, sheets[sheetIdx][idx][3]], dtype = np.integer))
				
				if rightImage < rightGap:
					sheets[sheetIdx].append(np.array([bottomGap, rightImage, sheets[sheetIdx][idx][2], rightGap - rightImage], dtype = np.integer))	
				
				if topImage < topGap:
					sheets[sheetIdx].append(np.array([topImage, leftGap, topGap - topImage, sheets[sheetIdx][idx][3]], dtype = np.integer))
				
				sheets[sheetIdx].pop(idx)
			#	idx += 1
			else:
				idx += 1
		
			sheets[sheetIdx] = uniqify_sheets(sheets[sheetIdx])
		
	return plan

def get_plan_area(imageLevels, plan):
	h = 0
	w = 0
	for img, loc in zip(imageLevels, plan):
		hh = loc[0] + img[0]
		ww = loc[1] + img[1]
		h = hh if hh > h else h
		w = ww if ww > w else w
	return h * w
	
def make_plan_visual(imageLevels, plan):
	h = 0
	w = 0
	for img, loc in zip(imageLevels, plan):
		hh = loc[0] + img[0]
		ww = loc[1] + img[1]
		h = hh if hh > h else h
		w = ww if ww > w else w
	print h, w
	
	plane = np.zeros((h, w), dtype = np.integer)
	
	idx = 1
	for img, loc in zip(imageLevels, plan):
		y = loc[0]
		x = loc[1]
		hh = img[0]
		ww = img[1]
		plane[y:y+hh, x:x+ww] = idx
		idx += 1
	
	plt.figure(figsize = (10, 10))
	plt.imshow(plane)
	plt.show()
	plt.colorbar()
	plt.gca().set_aspect('auto')
	
	return plane
	
if __name__ == "__main__":
	width = 690
	height = 490
	
	downscaleRate = 0.707
	
	minLen = 24
	
	w = width
	h = height
	imageLevels = []
	while w >= minLen and h >= minLen:
		imageLevels.append((h, w))
		w = int(w * downscaleRate + 0.5)
		h = int(h * downscaleRate + 0.5)
	
	print imageLevels
	
	plan = make_stitch_plan(imageLevels, height, 890)
	
	optPlan, minArea = plan_optimizer(imageLevels)
	
	print 'Optimized stitch plan:', optPlan
	print 'minimum area:', minArea
	
	make_plan_visual(imageLevels, optPlan)

	# TODO: read test image, resize image with different size, stitch the pyramid images into a large plane with the stitch plan
