/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *      Modified: Hasan Chowdhury
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void 
ParticleFilter::init (double x, double y, double theta, double std[])
{

    int                     ix;
    default_random_engine   gen;
    double                  std_x, std_y, std_theta;

    /*
     * Set the number of particles
     */
    num_particles = 100;
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    /*
     * generate random number wtih mean x and standard deviation of std_x; 
     */
    normal_distribution<double> dist_x(x, std_x);

    /*
     * generate random number wtih mean y and standard deviation of std_y;
     */
    normal_distribution<double> dist_y(y, std_y);

    /*
     * generate random number wtih mean theta and standard deviation of
     * std_theta; 
     */
    normal_distribution<double> dist_theta(theta, std_theta);
    
    /*
     * Initialize all particles at time t.
     * believe state at time t.
     *
     * Initialize all particles to first position (based on estimates of
     * x, y, theta and their uncertainties from GPS)
     * Add random Gaussian noise to each particle.
     */
    for (ix = 0; ix < num_particles; ix++ ) {
        Particle pt;
        pt.id = ix;
        pt.x = dist_x(gen);
        pt.y = dist_y(gen);
        pt.theta = dist_theta(gen);
        pt.weight = 1.0;
        particles.push_back(pt);
    }

    /*
     * Initialize each weight of weights vector to 1
     */
    weights.resize(num_particles, 1.0);

    /*
     * done initialization
     */
    is_initialized = true;
    cout<< "Particle filter is Initialized" <<endl;
}

void 
ParticleFilter::prediction (double delta_t, double std_pos[], double velocity,
                            double yaw_rate) {

    /*
     * Predict the particles at t+1 time.
     *
     * Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution and 
     * std::default_random_engine useful.
     * http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     * http://www.cplusplus.com/reference/random/default_random_engine/
     */

    int                     ix;
    default_random_engine   gen;
    double                  std_x, std_y, std_theta;
    double                  x1, x0, y1, y0, theta0;

    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    for (ix = 0; ix < num_particles; ix++ ) {
        x0 = particles[ix].x;
        y0 = particles[ix].y;
        theta0 = particles[ix].theta;
       
        /*
         * if yaw rate is zero or close to zero.
         * -no yaw acceleration
         */
        if (fabs(yaw_rate) < 0.0001) {
            x1 = x0 + velocity*delta_t*cos(theta0);
            y1 = y0 + velocity*delta_t*sin(theta0);
        }
        else {

            /*
             * with yaw acceleration
             */
            x1 = x0 + (velocity/yaw_rate)*(sin(theta0 + yaw_rate*delta_t)
                                                            - sin(theta0));
            y1 = y0 + (velocity/yaw_rate)*(cos(theta0) - cos(theta0 
                                                      + yaw_rate*delta_t));
            theta0 += yaw_rate*delta_t;
        }
        
        /*
         * add Gaussian noise into velocity and yaw rate
         */
        normal_distribution<double> dist_x(x1, std_x);
        normal_distribution<double> dist_y(y1, std_y);
        normal_distribution<double> dist_theta(theta0, std_theta);
        particles[ix].x =  dist_x(gen);
        particles[ix].y =  dist_y(gen);
        particles[ix].theta = dist_theta(gen);
    }

}

void 
ParticleFilter::dataAssociation (std::vector<LandmarkObs> predicted, 
                                 std::vector<LandmarkObs>& observations)
                                {

    /*
     * Find the predicted measurement that is closest to each observed
     * measurement and assign the observed measurement to this particular
     * landmark. 
     * NOTE: this method will NOT be called by the grading code.
     * But you will probably find it useful to implement this method and use 
     * it as a helper during the updateWeights phase.
     */
    LandmarkObs pred;
    int id;
    double euclidean_distance, min_assoc_distance;

    /*
     * Complexity O(m,n)
     * 
     * m = observed landmark
     * n = mapped landmark
     */ 
    for (int m = 0; m < observations.size(); m++) {
        LandmarkObs &obj = observations[m];
        
        /*
         * set the minimum association distance to 50m same as sensor range.
         * the difference between observed and mapped landmark should not be
         * more than 50m i.e. sensor range.
         */
        min_assoc_distance = 50;

        /*
         * always set to an invalid and if we find a closest match
         * this id will be set accordingly.
         */
        id = -1;
        for (int n = 0; n < predicted.size(); n++) {
            pred = predicted[n];

            /*
             * Match each pseudo range estimate to its closest observation
             * measurement; use Euclidean distance between two 2D points
             * to find the closest observation.
             */
            euclidean_distance = dist(obj.x, obj.y, pred.x, pred.y);
            if (euclidean_distance < min_assoc_distance) {
                min_assoc_distance = euclidean_distance;

                /*
                 * save the index of the closest predicted landmark
                 * in the id of the observed landmark, for
                 * a quick lookup later on.
                 */
                obj.id = n;
            }
        }
        
    }
}

void
ParticleFilter::updateWeights (double sensor_range, double std_landmark[],
                               const std::vector<LandmarkObs> &observations,
                               const Map &map_landmarks)
{
    
    /*
     * Update the weights of each particle using a mult-variate Gaussian
     * distribution. You can read more about this distribution
     * here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     * Your particles are located according to the MAP'S coordinate system.
     * You will need to transform between the two systems.
     * Keep in mind that this transformation requires both rotation AND
     * translation (but no scaling).
     *
     * The following is a good resource for the theory:
     * https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     * and the following is a good resource for the actual equation to
     * implement (look at equation 3.33)
     * http://planning.cs.uiuc.edu/node99.html
     */

    double x_part, y_part, theta;              /* particle coordinates in map
                                                * coordinate
                                                */
    double x_obs, y_obs;                       /* coordinates of landmark in
                                                * vehicle coordinates
                                                */
    double x_map, y_map;                       /* transformed landmarks in map
                                                * coordinate
                                                */
    double x, y;                               /*
                                                * landmarks in map coordinate
                                                */
    double dx, dy;                             /*
                                                * Distance between particles and
                                                * observed landmark in map coordinate
                                                */
    double distance, min_distance;
    double std_x, std_y;    
    int    observe_id, predict_id, ix, m, n;
    double gauss_norm, exponent;

    std_x = std_landmark[0];
    std_y = std_landmark[1];
    gauss_norm = (1/(2*M_PI*std_x*std_y));
    
    /*
     * Set the nearest neighbour distance to max sensor range
     * 
     */
    min_distance = sensor_range;

    for (ix = 0; ix < num_particles; ix++ ) {

        vector<LandmarkObs> t_plus_1_predic;
        vector<LandmarkObs> t_plus_1_measure;


        /*
         * particle in map coordinate
         */
        x_part = particles[ix].x;
        y_part = particles[ix].y;
        theta = particles[ix].theta;

        particles[ix].weight = 1.0;
        
        /* 
         * 1. Find the pseudo ranges within sensor range and store them in
         * vector t_plus_1_predic
         *
         * complexity O(m)
         * m = mapped landmark
         */
        for (m = 0; m < map_landmarks.landmark_list.size(); m++) {
            
            /*
             * landmark x,y in map coordinate
             */
            x = map_landmarks.landmark_list[m].x_f;
            y = map_landmarks.landmark_list[m].y_f;
            predict_id = map_landmarks.landmark_list[m].id_i;
            
            /*
             * Distance from particle to the nearest landmark within
             * sensor range
             */
            distance = sqrt((x - x_part)*(x - x_part) +
                                             (y - y_part)*(y - y_part));
            if (distance <= min_distance) {
                t_plus_1_predic.push_back(LandmarkObs{predict_id, x, y});
            } 
        }

        /*
         * 2. Transform the observation at t + 1 time into map coordinate 
         * and store them in vector t_plus_1_measure
         * complexity O(n)
         */
        for (n = 0; n < observations.size(); n++) {

            /*
             * observation in car coordinate
             */
            x_obs = observations[n].x;
            y_obs = observations[n].y;

            /*
             * transformed observation in map coordinate
             */
            x_map = x_part + cos(theta)*x_obs - sin(theta)*y_obs;
            y_map = y_part + sin(theta)*x_obs + cos(theta)*y_obs;

            /*
             * set the observed landmark into a invalid id here
             * i.e. -1
             */
            t_plus_1_measure.push_back(LandmarkObs{-1, x_map, y_map});
        }

        /*
         * 3. Match each pseudo range estimate at time t + 1 to its closest
         *    observation measurement and set the observation id accordingly
         */
        dataAssociation(t_plus_1_predic, t_plus_1_measure);

        /*
         * 4. For each pseudo range and observation measurement pair,
         *    calculate the corresponding probability 
         *    
         *    Assumption all observations are indipendent and the total
         *    probalility is the multiplication of all these individual
         *    probalility for this particles[ix]
         *
         *    complexity O(nm)
         */
        for (n = 0; n < t_plus_1_measure.size(); n++) {
            x_map = t_plus_1_measure[n].x;
            y_map = t_plus_1_measure[n].y;
            observe_id = t_plus_1_measure[n].id;
            
            /*
             * if a closet association match is found probability will be
             * hgher and if no close association found the probability will
             * be lower; resultant probability will be somewhere in between,
             * reflecting the overall belief of this observation model and
             * will be saved to each patrciles weight variable i.e.
             * particles[ix].weight;
             */
            if (t_plus_1_measure[n].id != -1) {

                /*
                 * Index of matching predict 
                 */
                int k = t_plus_1_measure[n].id;
                x_part = t_plus_1_predic[k].x;
                y_part = t_plus_1_predic[k].y;
            }

            dx = x_part - x_map;
            dy = y_part - y_map;

            exponent = dx*dx/(2*std_x*std_x) + dy*dy/(2*std_y*std_y);
            double poss = gauss_norm*exp(-exponent);
            
            particles[ix].weight *= poss;
        } // for each observation
        
        /*
         * Assign the observation model probability of ix_th particle into
         * the ix_th coordinate of weights vector.
         */
        weights[ix] = particles[ix].weight;
    } // for each particle
}

void
ParticleFilter::resample ()
{

    /*
     * Resample particles with replacement with probability proportional
     * to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     * http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */ 
    vector<Particle>              new_particles;
    default_random_engine         gen;
    uniform_int_distribution<int> dis(0, num_particles + 1);
    double                        max_weight;
    double                        beta;
    int                           index;
    
    beta = 0.0f;
    index = dis(gen);

#if 0
    cout<<"Weight ";
    for (int k = 0; k < num_particles; k++) {
        cout<<weights[k]<<" ";
    }
    cout<<endl;
#endif
    max_weight = *max_element(weights.begin(), weights.end());
    
    //cout<<"max_weight "<<max_weight<<endl;

    /*
     * Resampling based on the Wheel model
     */
    for (int i = 0; i < num_particles; i++) {
        beta += dis(gen)*2.0*max_weight;  
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;
}

void
ParticleFilter::SetAssociations (Particle& particle,
                                 const std::vector<int>& associations, 
                                 const std::vector<double>& sense_x,
                                 const std::vector<double>& sense_y)
{

    /*
     * particle: the particle to assign each listed association, and
     * association's (x,y) world coordinates mapping to
     * associations: The landmark id that goes along with each listed
     *               association
     * sense_x: the associations x mapping already converted to world
     *          coordinates
     * sense_y: the associations y mapping already converted to world
     * coordinates
     */

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string
ParticleFilter::getAssociations (Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
