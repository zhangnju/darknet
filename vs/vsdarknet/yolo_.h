/*
 * File Name: yolo_.c
 * Description:
 * Author: Lili Zhao <zhao.lili@xiaoyi.com>
 * Date: Aug., 2016
*/

#ifndef _YOLO_H_
#define _YOLO_H_

#include "parser.h"
#include "utils.h"

#define	_MAX_NUM_OF_OBJECTS		40
typedef struct _yolo_grid
{
	int		grids;		// numbebr of grid
	int		bbs;		// number of bounding box
	int		classes;	// number of classes

	box		*boxes;
	float	**probs;
}yoloGrid;

typedef struct _yolo_pos
{
	int	_left;
	int	_right;
	int	_top;
	int	_bottom;
}yoloPos;

typedef struct _yolo_object
{
	char	_classname[200];
	float	_prob;

	yoloPos	_pos;
}yoloObject;

typedef struct _yolo_predictions
{
	int _count;			//number of detected objects
	yoloObject	_objects[_MAX_NUM_OF_OBJECTS];
}yoloPredictions;

typedef struct _context_param_yolo
{
	network		_net;
	yoloGrid	_grid;

	int	_nwidth;		//input image width for network
	int	_nheight;		//input image height for network

	int _sqrt;	
	float _nms;
}context_param_yolo_t;

/*
 * create YOLO network 
*/
void createYoloNetwork(context_param_yolo_t *yoloctx, char* cfgfile, char* weightfile);

/*
 * destroy YOLO network
*/
void destroyYoloNetwork(context_param_yolo_t *yoloctx);

/*
 * do prediction
 * @param[in]: yoloctx, context
 * @param[in]: filename, input picture
 * @param[in]: thresh, threshold for probability x confidence level
 * @param[out]: predictions, store detected objects
 */
void yoloPredict(context_param_yolo_t *yoloctx, char *filename, float thresh, yoloPredictions *predictions);

#endif