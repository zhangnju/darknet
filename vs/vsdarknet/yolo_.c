/*
 * File Name: yolo_.c
 * Description: 
 * Author: Lili Zhao <zhao.lili@xiaoyi.com>
 * Date: Aug., 2016
*/

#include <string.h>
#include "parser.h"
#include "detection_layer.h"
#include "network.h"
#include "yolo_.h"

#define	OBJECT_CLASSES	20
static char *class_names[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
static void convertResults(int iwidth, int iheight, int num, float thresh, box *boxes, float **probs, char **names, int classes, yoloPredictions *predictions)
{
	int count;	
	int i;

	count = 0;
	for (i = 0; i < num; ++i){
		int class = max_index(probs[i], classes);
		float prob = probs[i][class];
		if (prob > thresh){
			if (count >= _MAX_NUM_OF_OBJECTS)
			{
				printf("Detected objects reaches the maximum number %d.\n", _MAX_NUM_OF_OBJECTS);
				break;
			}

			printf("%s: %.0f%%\n", names[class], prob * 100);
			int offset = class * 1 % classes;
			box b = boxes[i];

			int left = (b.x - b.w / 2.)*iwidth;
			int right = (b.x + b.w / 2.)*iwidth;
			int top = (b.y - b.h / 2.)*iheight;
			int bot = (b.y + b.h / 2.)*iheight;

			if (left < 0) left = 0;
			if (right > iwidth - 1) right = iwidth - 1;
			if (top < 0) top = 0;
			if (bot > iheight - 1) bot = iheight - 1;

			yoloObject obj;
			sprintf(obj._classname, "%s\0", names[class]);
			obj._prob = prob;
			obj._pos._left	= left;
			obj._pos._right = right;
			obj._pos._top	= top;
			obj._pos._bottom = bot;

			predictions->_objects[count++] = obj;
		}
	}

	predictions->_count = count;
}

static void convertDetections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
	int i, j, n;
	//int per_cell = 5*num+classes;
	for (i = 0; i < side*side; ++i){
		int row = i / side;
		int col = i % side;
		for (n = 0; n < num; ++n){
			int index = i*num + n;
			int p_index = side*side*classes + i*num + n;
			float scale = predictions[p_index];
			int box_index = side*side*(classes + num) + (i*num + n) * 4;
			boxes[index].x = (predictions[box_index + 0] + col) / side * w;
			boxes[index].y = (predictions[box_index + 1] + row) / side * h;
			boxes[index].w = pow(predictions[box_index + 2], (square ? 2 : 1)) * w;
			boxes[index].h = pow(predictions[box_index + 3], (square ? 2 : 1)) * h;
			for (j = 0; j < classes; ++j){
				int class_index = i*classes;
				float prob = scale*predictions[class_index + j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (only_objectness){
				probs[index][0] = scale;
			}
		}
	}
}

static void resetData(context_param_yolo_t *yoloctx)
{
	float	**probs = yoloctx->_grid.probs;
	box		*boxes = yoloctx->_grid.boxes;

	int side = yoloctx->_grid.grids;
	int classes = yoloctx->_grid.classes;
	int bbs = yoloctx->_grid.bbs;
	int i, j;
	for (j = 0; j < side * side * bbs; j++)
	{
		boxes[j].h = boxes[j].w = boxes[j].x = boxes[j].y = 0;
		for (i = 0; i < classes; i++)
		{
			probs[j][i] = 0.0f;
		}
	}
}

/*
 * @param[in]: ctx
*/
void createYoloNetwork(context_param_yolo_t *yoloctx, char* cfgfile, char* weightfile)
{
	printf("Create YOLO network\n");
	network net = parse_network_cfg(cfgfile);
	if (weightfile)
	{
		load_weights(&net, weightfile);
	}

	set_batch_network(&net, 1);
	detection_layer l = net.layers[net.n - 1];

	yoloGrid grid;
	grid.grids = l.side;
	grid.bbs = l.n;
	grid.classes = l.classes;

	box		*boxes = malloc(l.side * l.side * l.n * sizeof(box));
	float	**probs = malloc(l.side * l.side * l.n * sizeof(float *));
	for (int j = 0; j < l.side * l.side * l.n; j++)
	{
		probs[j] = malloc(l.classes*sizeof(float *));
	}
	
	yoloctx->_net	= net;
	yoloctx->_grid	= grid;
	yoloctx->_grid.boxes	= boxes;
	yoloctx->_grid.probs	= probs;

	yoloctx->_nwidth	= net.w;
	yoloctx->_nheight	= net.h;

	yoloctx->_sqrt		= l.sqrt;
	yoloctx->_nms		= .5f;		// non maximal suppression
}

void destroyYoloNetwork(context_param_yolo_t *yoloctx)
{
	printf("Destroy YOLO network\n");
	int side	= yoloctx->_grid.grids;
	int bb		= yoloctx->_grid.bbs;

	box *boxes = yoloctx->_grid.boxes;
	if (boxes)
	{
		free(boxes);
	}

	float** probs = yoloctx->_grid.probs;
	for (int j = 0; j < side * side * bb; j++)
	{
		free(probs[j]);
	}
	free(probs);

	// todo: free network

	yoloctx->_grid.boxes = NULL;
	yoloctx->_grid.probs = NULL;
}

/*
 * do prediction
  *@param[in]: yoloctx, context
  *@param[in]: filename, input picture
  *@param[in]: thresh, threshold for probability x confidence level
  *@param[out]: predictions, store detected objects
*/
void yoloPredict(context_param_yolo_t *yoloctx, char *filename, float thresh, yoloPredictions *predictions)
{
	printf("YOLO predict\n");
	int	nwidth	= yoloctx->_nwidth;
	int nheight = yoloctx->_nheight;
	int side	= yoloctx->_grid.grids;
	int classes = yoloctx->_grid.classes;
	int bbs		= yoloctx->_grid.bbs;
	int sqrt	= yoloctx->_sqrt;
	float nms		= yoloctx->_nms;

	image im	= load_image_color(filename, 0, 0);
	image sized = resize_image(im, nwidth, nheight);
	
	resetData(yoloctx);

	float *x = sized.data;
	float *fpredictions = network_predict(yoloctx->_net, x);

	float	**probs = yoloctx->_grid.probs;
	box		*boxes = yoloctx->_grid.boxes;

	convertDetections(fpredictions, classes, bbs, sqrt, side, 1, 1, thresh, probs, boxes, 0); 
	if (nms) do_nms_sort(boxes, probs, side*side*bbs, classes, nms);
	convertResults(im.w, im.h, side*side*bbs, thresh, boxes, probs, class_names, 20, predictions);
	
	//free(predictions);
	free_image(sized);
	free_image(im);
}