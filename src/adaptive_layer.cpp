#include <adaptive_social_layers/adaptive_layer.h>
#include <math.h>
#include <angles/angles.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>


PLUGINLIB_EXPORT_CLASS(adaptive_social_layers::AdaptiveLayer, costmap_2d::Layer)

using costmap_2d::NO_INFORMATION;
using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::FREE_SPACE;

double gaussianPerson(double x, double y, double x0, double y0, double A, double varx, double vary, double skew){
    double dx = x-x0, dy = y-y0;
    double h = sqrt(dx*dx+dy*dy);
    double haux = sqrt(0.45*0.45);
    double angle = atan2(dy,dx);
    double angleaux = atan2(1,0);
    double mx = cos(angle-skew) * h;
    double my = sin(angle-skew) * h;
    double mxaux = cos(angleaux) * haux;
    double myaux = sin(angleaux) * haux;
    double f1 = pow(mx, 2.0)/(2.0 * varx),
           f2 = pow(my, 2.0)/(2.0 * vary);
    double f1aux = pow(mxaux, 2.0)/(2.0 * varx),
           f2aux = pow(myaux, 2.0)/(2.0 * vary);
    double Aux = 254/(exp(-(f1aux + f2aux)));
    double gauss = A * exp(-(f1 + f2));
    if(gauss > 254)
    {
        return 254;
    }
    else if (gauss < 0)
    {
        return 0;
    }
    else
    {
        return (int) (gauss + 0.5);
    }
}

double gaussian(double x, double y, double x0, double y0, double A, double varx, double vary, double skew){
    double dx = x-x0, dy = y-y0;
    double h = sqrt(dx*dx+dy*dy);
    double angle = atan2(dy,dx);
    double mx = cos(angle-skew) * h;
    double my = sin(angle-skew) * h;
    double f1 = pow(mx, 2.0)/(2.0 * varx),
           f2 = pow(my, 2.0)/(2.0 * vary);
    double gauss = A * exp(-(f1 + f2));
    if(gauss > 254)
    {
        return 254;
    }
    else if (gauss < 0)
    {
        return 0;
    }
    else
    {
        return (int) (gauss+0.5);
    }
}

double get_radius(double cutoff, double A, double var){
    return sqrt(-2*var * log(cutoff/A) );
}

double distance(double x1, double y1, double x2, double y2){
    double square_difference_x = (x2 - x1) * (x2 - x1);
    double square_difference_y = (y2 - y1) * (y2 - y1);
    double sum = square_difference_x + square_difference_y;
    double value = sqrt(sum);
    return value;
}



namespace adaptive_social_layers
{
    void AdaptiveLayer::onInitialize()
    {
        SocialLayer::onInitialize();
        ros::NodeHandle nh("~/" + name_), g_nh;
        server_ = new dynamic_reconfigure::Server<AdaptiveLayerConfig>(nh);
        f_ = boost::bind(&AdaptiveLayer::configure, this, _1, _2);
        server_->setCallback(f_);
    }

    void AdaptiveLayer::updateBoundsFromPeople(double* min_x, double* min_y, double* max_x, double* max_y)
    {
        std::list<group_msgs::Person>::iterator p_it;

        for(p_it = transformed_people_.begin(); p_it != transformed_people_.end(); ++p_it){
            group_msgs::Person person = *p_it;

            double point;
            if (!person.ospace) 
                point = std::max(person.sx,person.sy);
            else
                point = std::max(person.sx,person.sy);

            *min_x = std::min(*min_x, person.position.x - point + 2 );
            *min_y = std::min(*min_y, person.position.y - point + 2 );
            *max_x = std::max(*max_x, person.position.x + point + 2 );
            *max_y = std::max(*max_y, person.position.y + point + 2 );

        }
    }

    void AdaptiveLayer::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j){
        boost::recursive_mutex::scoped_lock lock(lock_);
        if(!enabled_) return;

        if( people_list_.people.size() == 0 )
          return;
        if( cutoff_ >= amplitude_)
            return;

        std::list<group_msgs::Person>::iterator p_it;
        costmap_2d::Costmap2D* costmap = layered_costmap_->getCostmap();
        double res = costmap->getResolution();
        int size = transformed_people_.size();


        for(p_it = transformed_people_.begin(); p_it != transformed_people_.end(); ++p_it){
            group_msgs::Person person = *p_it;
            double angle = person.orientation;

            double var;
            double varp;
            double base;
            double point ;
            if (!person.ospace){ 

                base = std::max(person.sx,person.sy) + 2  ;
                point = std::max(person.sx,person.sy) + 2;
            }

            else{
                base = std::max(person.sx,person.sy) + 2;
                point = std::max(person.sx,person.sy) + 2;
            }

            unsigned int width = std::max(1, int( (base + point) / res )),
                          height = std::max(1, int( (base + point) / res ));

            double cx = person.position.x, cy = person.position.y;

            double ox, oy;
            if(sin(angle)>0)
                oy = cy - base;
            else
                oy = cy + (point-base) * sin(angle) - base;

            if(cos(angle)>=0)
                ox = cx - base;
            else
                ox = cx + (point-base) * cos(angle) - base;


            int dx, dy;
            costmap->worldToMapNoBounds(ox, oy, dx, dy);

            int start_x = 0, start_y=0, end_x=width, end_y = height;
            if(dx < 0)
                start_x = -dx;
            else if(dx + width > costmap->getSizeInCellsX())
                end_x = std::max(0, (int)costmap->getSizeInCellsX() - dx);

            if((int)(start_x+dx) < min_i)
                start_x = min_i - dx;
            if((int)(end_x+dx) > max_i)
                end_x = max_i - dx;

            if(dy < 0)
                start_y = -dy;
            else if(dy + height > costmap->getSizeInCellsY())
                end_y = std::max(0, (int) costmap->getSizeInCellsY() - dy);

            if((int)(start_y+dy) < min_j)
                start_y = min_j - dy;
            if((int)(end_y+dy) > max_j)
                end_y = max_j - dy;

            double bx = ox + res / 2,
                   by = oy + res / 2;
            for(int i=start_x;i<end_x;i++){
                for(int j=start_y;j<end_y;j++){
                    unsigned char old_cost = costmap->getCost(i+dx, j+dy);
                    if(old_cost == costmap_2d::NO_INFORMATION)
                    continue;

                    double x = bx+i*res, y = by+j*res;
                    double ma = atan2(y-cy,x-cx);
                    double diff = angles::shortest_angular_distance(angle, ma);
                    double a;

                    
                    // Convert personal space parameters to ros gaussian parameters for a fixed amplitude and cutoff 
                    double sx = (pow(person.sx, 2) / (log(cutoff_/amplitude_))/ (-2));
                    double sy = (pow(person.sy, 2) / (log(cutoff_/amplitude_))/ (-2));
                    double sx_back = (pow(person.sx_back, 2) / (log(cutoff_/amplitude_))/ (-2));
                    double sy_right = (pow(person.sy_right, 2) / (log(cutoff_/amplitude_))/ (-2));


                    if(person.ospace){
                        if(fabs(diff)<M_PI/2)
                            a = gaussian(x,y,cx,cy,amplitude_, sx, sy,person.orientation);
                        else
                            a = gaussian(x,y,cx,cy,amplitude_, sx_back, sy,person.orientation);
                    }

                    else {
                        /* if (distance(x,y,cx,cy) <= HUMAN_Y/2 ){
                            double cost = costmap_2d::LETHAL_OBSTACLE;
                            costmap->setCost(i+dx, j+dy, cost);
                            a = costmap_2d::LETHAL_OBSTACLE;
                        } //Mark person body area as lethal 

                        else{ // Compute gaussian value of the cell

                            if(fabs(diff)<M_PI/2)
                                a = gaussian(x,y,cx,cy,amplitude_, sx, sy,person.orientation);
                            else
                                a = gaussian(x,y,cx,cy,amplitude_, sx_back, sy,person.orientation);
            
                        } */
                        
                        if(fabs(diff)<M_PI/2)
                        {
                            //right
                            if (diff < 0)
                                a = gaussianPerson(x,y,cx,cy,amplitude_, sx, sy_right,person.orientation);
                            else 
                                a = gaussianPerson(x,y,cx,cy,amplitude_, sx, sy,person.orientation);
                        }
                        else
                        {
                            if (diff < 0)
                                a = gaussianPerson(x,y,cx,cy,amplitude_, sx_back, sy_right,person.orientation);
                            else
                                a = gaussianPerson(x,y,cx,cy,amplitude_, sx_back, sy,person.orientation);
                        }
    
                    }
                
                    if(a < cutoff_)
                        continue;
                    unsigned char cvalue = (unsigned char) a;
                    costmap->setCost(i+dx, j+dy, std::max(cvalue, old_cost));

              }
            }


        }
    }

    void AdaptiveLayer::configure(AdaptiveLayerConfig &config, uint32_t level) {
        cutoff_ = config.cutoff;
        amplitude_ = config.amplitude;
        factor_ = config.factor;
        people_keep_time_ = ros::Duration(config.keep_time);
        enabled_ = config.enabled;
    }


};