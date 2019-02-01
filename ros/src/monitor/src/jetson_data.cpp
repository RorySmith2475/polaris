#include <ros/ros.h>
#include "std_msgs/String.h"
#include "monitor/jetson_data.h"

#include <string>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <math.h>

void update_values(std::string input)
{
    float values[200];
    int index = 0;
    float temp = 0.0;
    bool found_num = false;
    int period_found = 0;


    for(int i=0; i< input.length(); i++){
        char c = input[i];
        if(isdigit(c)){
            if(period_found > 0){
                temp += (float)(c - '0')/pow(10.0, period_found);
                period_found++;
            }
            else{
                temp *= 10.0;
                temp += (float)(c - '0');
            }
            found_num = true;
        }
        else if(c == '.'){
            period_found++;
        }
        else if(found_num){
            values[index] = temp;
            index++;
            temp = 0.0;
            found_num = false;
            period_found = 0;
        }
        else{
            temp = 0.0;
            found_num = false;
            period_found = 0;
        }
        if(i + 2 <= input.length() && c == 'o') {
            if((input[i+1] == 'f') && (input[i+2] == 'f')) {
                values[index++] = -1;
                values[index++] = -1;
            }
        }
    }
    
    msg.RAM_N = (int)values[0];
    msg.RAM_D = (int)values[1];
    msg.CPU_usage_1 = (int)values[4];
    msg.CPU_usage_2 = (int)values[6];
    msg.CPU_usage_3 = (int)values[8];
    msg.CPU_usage_4 = (int)values[10];
    msg.CPU_usage_5 = (int)values[12];
    msg.CPU_usage_6 = (int)values[14];
    msg.BCPU_temp = values[16];
    msg.MCPU_temp = values[17];
    msg.GPU_temp = values[18];
    msg.PLL_temp = values[19];
    msg.Tboard_temp = values[20];
    msg.Tdiode_temp = values[21];
    msg.PMIC_temp = values[22];
    msg.VDD_IN_N = (int)values[24];
    msg.VDD_IN_D = (int)values[25];
    msg.VDD_CPU_N = (int)values[26];
    msg.VDD_CPU_D = (int)values[27];
    msg.VDD_GPU_N = (int)values[28];
    msg.VDD_GPU_D = (int)values[29];
    msg.VDD_SOC_N = (int)values[30];
    msg.VDD_SOC_D = (int)values[31];
    msg.VDD_WIFI_N = (int)values[32];
    msg.VDD_WIFI_D = (int)values[33];
    msg.VDD_DDR_N = (int)values[34];
    msg.VDD_DDR_D = (int)values[35];

    for(int i = 0; i < index; i++){
        std::cout << values[i] << std::endl;
    }
}

int main(int argc, char **argv)
 {
    ros::init(argc, argv, "talker");
    ros::NodeHandle nh;
    ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter",5000);
    ros::Rate loop_rate(10);

    std::array<char, 1024> buffer;
    std::string input;
    std::string prev_input = "";
    FILE* pipe = popen("/home/rorysmith/Desktop/filetoexe", "r");

    if(!pipe) msg.ERROR = 1;
    else msg.ERROR = 0;

    while(ros::ok()) {
        if(fgets(buffer.data(), 1024, pipe) != NULL) {
            input = buffer.data();
            std::cout << "\n" << input << "\n";//remove
        }
        if(prev_input != input) {
            update_values(input);
        } 
        prev_input = input;

        std_msgs::String msg;
        std::stringstream ss;
        msg.data = ss.str();
        
        chatter_pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
    pclose(pipe);
    return 0;   
}