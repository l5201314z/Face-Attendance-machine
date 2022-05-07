/**
 * Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * @author baidu aip
 */

#ifndef __AIP_BODYANALYSIS_H__
#define __AIP_BODYANALYSIS_H__

#include "base/base.h"

namespace aip {

    class Bodyanalysis: public AipBase
    {
    public:

    
        std::string _body_analysis =
            "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_analysis";
        
        std::string _body_attr =
            "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_attr";
        
        std::string _body_num =
            "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_num";
        
        std::string _gesture =
            "https://aip.baidubce.com/rest/2.0/image-classify/v1/gesture";
        
        std::string _body_seg =
            "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_seg";
        

        Bodyanalysis(const std::string & app_id, const std::string & ak, const std::string & sk): AipBase(app_id, ak, sk)
        {
        }
        
        /**
         * body_analysis
         * 对于输入的一张图片（可正常解码，且长宽比适宜），**检测图片中的所有人体，输出每个人体的14个主要关键点，包含四肢、脖颈、鼻子等部位，以及人体的坐标信息和数量**。
         * @param image 图像文件二进制内容，可以使用aip::get_file_content函数获取
         * options 可选参数:
         */
        Json::Value body_analysis(
            std::string const & image,
            const std::map<std::string, std::string> & options)
        {
            std::map<std::string, std::string> data;
            
            data["image"] = base64_encode(image.c_str(), (int) image.size());

            std::copy(options.begin(), options.end(), std::inserter(data, data.end()));

            Json::Value result =
                this->request(_body_analysis, null, data, null);

            return result;
        }
        
        /**
         * body_attr
         * 对于输入的一张图片（可正常解码，且长宽比适宜），**检测图像中的所有人体并返回每个人体的矩形框位置，识别人体的静态属性和行为，共支持20余种属性，包括：性别、年龄阶段、衣着（含类别/颜色）、是否戴帽子、是否戴眼镜、是否背包、是否使用手机、身体朝向等**。
         * @param image 图像文件二进制内容，可以使用aip::get_file_content函数获取
         * options 可选参数:
         * type gender,<br>age,<br>lower_wear,<br>upper_wear,<br>headwear,<br>glasses,<br>upper_color,<br>lower_color,<br>cellphone,<br>upper_wear_fg,<br>upper_wear_texture,<br>lower_wear_texture,<br>orientation,<br>umbrella,<br>bag,<br>smoke,<br>vehicle,<br>carrying_item,<br>upper_cut,<br>lower_cut,<br>occlusion &#124; 1）可选值说明：<br>gender-性别，<br>age-年龄阶段，<br>lower_wear-下身服饰，<br>upper_wear-上身服饰，<br>headwear-是否戴帽子，<br>glasses-是否戴眼镜，<br>upper_color-上身服饰颜色，<br>lower_color-下身服饰颜色，<br>cellphone-是否使用手机，<br>upper_wear_fg-上身服饰细分类，<br>upper_wear_texture-上身服饰纹理，<br>orientation-身体朝向，<br>umbrella-是否撑伞；<br>bag-背包,<br>smoke-是否吸烟,<br>vehicle-交通工具,<br>carrying_item-是否有手提物,<br>upper_cut-上方截断,<br>lower_cut-下方截断,<br>occlusion-遮挡<br>2）type 参数值可以是可选值的组合，用逗号分隔；**如果无此参数默认输出全部20个属性**
         */
        Json::Value body_attr(
            std::string const & image,
            const std::map<std::string, std::string> & options)
        {
            std::map<std::string, std::string> data;
            
            data["image"] = base64_encode(image.c_str(), (int) image.size());

            std::copy(options.begin(), options.end(), std::inserter(data, data.end()));

            Json::Value result =
                this->request(_body_attr, null, data, null);

            return result;
        }
        
        /**
         * body_num
         * 对于输入的一张图片（可正常解码，且长宽比适宜），**识别和统计图像当中的人体个数（静态统计，暂不支持追踪和去重）**。
         * @param image 图像文件二进制内容，可以使用aip::get_file_content函数获取
         * options 可选参数:
         * area 特定框选区域坐标，逗号分隔，如‘x1,y1,x2,y2,x3,y3...xn,yn'，默认尾点和首点相连做闭合，**此参数为空或无此参数默认识别整个图片的人数**
         * show 是否输出渲染的图片，默认不返回，**选true时返回渲染后的图片(base64)**，其它无效值或为空则默认false
         */
        Json::Value body_num(
            std::string const & image,
            const std::map<std::string, std::string> & options)
        {
            std::map<std::string, std::string> data;
            
            data["image"] = base64_encode(image.c_str(), (int) image.size());

            std::copy(options.begin(), options.end(), std::inserter(data, data.end()));

            Json::Value result =
                this->request(_body_num, null, data, null);

            return result;
        }
        
        /**
         * gesture
         * 识别图片中的手势类型，返回手势名称、手势矩形框、概率分数，可识别22种手势，支持动态手势识别，适用于手势特效、智能家居手势交互等场景；支持的22类手势列表：手指、掌心向前、拳头、OK、祈祷、作揖、作别、单手比心、点赞、diss、rock、掌心向上、双手比心（3种）、数字（7种）。
         * @param image 图像文件二进制内容，可以使用aip::get_file_content函数获取
         * options 可选参数:
         */
        Json::Value gesture(
            std::string const & image,
            const std::map<std::string, std::string> & options)
        {
            std::map<std::string, std::string> data;
            
            data["image"] = base64_encode(image.c_str(), (int) image.size());

            std::copy(options.begin(), options.end(), std::inserter(data, data.end()));

            Json::Value result =
                this->request(_gesture, null, data, null);

            return result;
        }
        
        /**
         * body_seg
         * 对于输入的一张图片（可正常解码，且长宽比适宜），**识别人体的轮廓范围，与背景进行分离，适用于拍照背景替换、照片合成、身体特效等场景。输入正常人像图片，返回分割后的二值结果图和分割类型（目前仅支持person）**
         * @param image 图像文件二进制内容，可以使用aip::get_file_content函数获取
         * options 可选参数:
         */
        Json::Value body_seg(
            std::string const & image,
            const std::map<std::string, std::string> & options)
        {
            std::map<std::string, std::string> data;
            
            data["image"] = base64_encode(image.c_str(), (int) image.size());

            std::copy(options.begin(), options.end(), std::inserter(data, data.end()));

            Json::Value result =
                this->request(_body_seg, null, data, null);

            return result;
        }
        

    };
}
#endif