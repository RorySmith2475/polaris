#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cmath>

class detection
{
public:
    detection();
    ~detection();
    bool update();
    bool camera_detected();
    int getX();
    int getY();

private:
    cv::Point find_best_point(std::vector<point_data> points);
    void add_point(std::vector<point_data> &points, cv::Point point, int radius);
    void remove_point(std::vector<point_data> &points);

    struct point_data {
        cv::Point p;
        uint8_t n;
    };

    cv::Point best_point;
    std::vector<point_data> points;
    cv::Mat frame;
    cv::VideoCapture src;
    cv::CascadeClassifier cascade;

    bool camera_found;
    const uint16_t radius;
    const uint8_t reset_time;
    uint8_t iteration;

}

detection::detection(int camera_name, std::string cascade_name, uint16_t r) : radius(r), reset_time(6) // TODO look into camera_name type and reset_time init
{
    src(camera_name);
    if((src.isOpened()) camera_found = true;
    else camera_found = false;

    object_cascade.load(cascade_name);
    iteration = 0;
}

detection::~detection() // TODO
{}

bool detection::update()
{
    if(!camera_found) return false;
    if(!src.read(frame)) return false;

    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    // Detects objects and stores their location in locations
    std::vector<cv::Rect> locations;
    object_cascade.detectMultiScale(frame_gray, locations, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    if(locations.empty()) return false;

    for(uint8_t i=0; i<locations.size(); i++)
    {
        cv::Point curr_point(object[0].x + object[0].width/2, object[0].y + object[0].height/2);
        add_point(points, curr_point, radius);
        best_point = find_point(points);
    }

    iteration++;
    if(iteration >= reset_time) {
        remove_point();
        iteration = 0;
    }
    
    return true;
}

bool detection::camera_detected() { return camera_found; }

int detection::getX() { return best_point.x; }

int detection::getY() { return best_point.y; }

/**
 * Iterates through points to find the point with the largest n. If n is
 * not greater than 3, the point (0,0) is returned.
 */
cv::Point detection::find_best_point(std::vector<point_data> points)
{
    uint8_t max = 0;
    uint8_t max_index = 0;
    for(int i=0; i<points.size(); i++)
    {
        if(points.at(i).n > max) {
            max = points.at(i).n;
            max_index = i;
        }
    }
    if(max > 1) return points.at(max_index).p;
    return {0,0};
}

/**
 * Iterates through points, adding one to n if the current point is within
 * radius distance. P is then updated to the current point.
 * 
 * If the current point is not within radius of any point in points,
 * a new index is created for it.  
 */
void detection::add_point(std::vector<point_data> &points, cv::Point point, uint16_t radius)
{
    for(int i=0; i<points.size(); i++)
    {
        if(sqrt(pow(point.x - points.at(i).p.x, 2) + pow(point.y - points.at(i).p.y, 2)) <= radius) {
            points.at(i).p.x = point.x;
            points.at(i).p.y = point.y;
            points.at(i).n = (points.at(i).n < 6) ? points.at(i).n + 1 : 6;
            return;
        }
    }
    points.push_back({point, 1});
}

/**
 * Iterates through points and removes one from every n in point_data.
 * If n == 0 in any point_data, remove the data from the vector points. 
 */
void detection::remove_point(std::vector<point_data> &points)
{
    for(int i=0; i<points.size(); i++)
    {
        points.at(i).n = points.at(i).n - 1;
        if(points.at(i).n == 0) {
            points.erase(points.begin() + i);
            i--;
        }
    }
}