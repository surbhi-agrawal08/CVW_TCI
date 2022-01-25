"""The file contains the code for the Crossing VRU Warning module.
"""
import math
import json
import numpy as np


class CVW(object):

    """The class to manage our DASHCAM warning for Crossing VRU Warning system.
    
    Attributes:
        AREA_PARAM (TYPE): Description
        AREA_TH (TYPE): Description
        CENTROID_TH (TYPE): Description
        CVW_FUNCTION_ACTIVE (TYPE): Description
        DIST_PARAM (TYPE): Description
        FOI_ANGLE (TYPE): Description
        INTERSECTION_SPEED_TH (TYPE): Description
        N_SECTORS (TYPE): Description
        number_of_frames (int): Description
        pedestrianAnalysis_object (list): Description
        VERTEX_TH (TYPE): Description
        WTS (TYPE): Description
        YAW_TH (TYPE): Description
    """
    
    def __init__(self, cvw_config):
        """Initialize the CVW class.
        
        Args:
            cvw_config (dict): The set of parameters for CVW passed from the config file.
        """
        self.init_config(cvw_config)
        self.FOI_ANGLE = (2 * np.pi / 5)
        self.number_of_frames = 10
        self.pedestrianAnalysis_object = []  # [{}]*self.number_of_frames

    def init_config(self, cvw_config):
        """Initialize the set of internal CVW parameters
        
        Args:
            cvw_config (dict): The set of parameters for CVW passed from the config file.
        """
        self.CVW_FUNCTION_ACTIVE = cvw_config['CVW_FUNCTION_ACTIVE']
        self.INTERSECTION_SPEED_TH = cvw_config['INTERSECTION_SPEED_TH']
        self.YAW_TH = cvw_config['YAW_TH']
        self.WTS = cvw_config['WTS']
        self.CENTROID_TH = cvw_config['CENTROID_TH']
        self.AREA_TH = cvw_config['AREA_TH']
        self.VERTEX_TH = cvw_config['VERTEX_TH']
        self.DIST_PARAM = cvw_config['DIST_PARAM']
        self.AREA_PARAM = cvw_config['AREA_PARAM']
        self.N_SECTORS = cvw_config['N_SECTORS']
    
    def check_cvw_active(self):
        """Check if we need to raise the CVW alert or not.
        
        Returns:
            bool: Returns True/False based on simple logic checks.
        """
        return self.CVW_FUNCTION_ACTIVE

    def raise_alert(self, crosswalks, objects, ego_twist):
        """The main function which gives out the output of this module used for raising the alert.
        
        Args:
            crosswalks (TYPE): Description
            objects (list): List of objects passed from each recogntion frame to the HMI Manager.
            ego_twist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        gyro_z_val = ego_twist.angular[2]
        ego_speed = math.dist(ego_twist.linear, [0, 0, 0])

        turn = self.predict_turn(gyro_z_val)
        
        cw_scores = self.get_CW_scores_IMU(
            cw_list=crosswalks,
            turn=turn,
            ego_speed=ego_speed,
        )

        ped_result = self.pedestrian_analysis(crosswalks, objects, cw_scores)

        CVW_output = [{} for i in range(len(ped_result))]
        for idx, result in enumerate(ped_result):
            CVW_output[idx]["ped_id"] = result["ped_id"]
            
            if result["IS_CROSSWALK_PROXIMITY"] == 2:
                CVW_output[idx]["CROSSING_INTENTION"] = True
            
            elif result["IS_CROSSWALK_PROXIMITY"] == 1:
                if result["IS_APPROACHING_CROSSWALK"] == 2:
                    CVW_output[idx]["CROSSING_INTENTION"] = True
                elif result["IS_APPROACHING_CROSSWALK"] == 1:
                    CVW_output[idx]["CROSSING_INTENTION"] = (
                        True
                        if (
                            result["PEDESTRIAN_FACE_DIRECTION"] > 0
                            or result["PEDESTRIAN_BODY_DIRECTION"] > 0
                        )
                        else False
                    )
                else:
                    CVW_output[idx]["CROSSING_INTENTION"] = False
            
            else:
                CVW_output[idx]["CROSSING_INTENTION"] = False

        return CVW_output

    def getMinMaxAngles(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        min_angle = np.pi
        max_angle = 0

        for point in points:
            angle = np.arctan2(point["y"], point["x"]) + np.pi / 2
            if angle > np.pi or angle < 0:
                continue
            if angle < min_angle:
                min_angle = angle
            if angle > max_angle:
                max_angle = angle

        return min_angle, max_angle

    def getMinMaxSectors(self, points, sector_angle):
        """Summary
        
        Args:
            points (TYPE): Description
            sector_angle (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        min_angle, max_angle = self.getMinMaxAngles(points)
        return min_angle // sector_angle, max_angle // sector_angle

    def getDistLS(self, p1, p2):
        """Summary
        
        Args:
            p1 (TYPE): Description
            p2 (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        v10 = np.array([-p1["x"], -p1["y"]])
        v12 = np.array([p2["x"] - p1["x"], p2["y"] - p1["y"]])

        r = v10.dot(v12) / (np.linalg.norm(v12) ** 2)

        if r < 0:
            return np.linalg.norm(v10)
        elif r > 1:
            return np.linalg.norm(v12 - v10)
        else:
            return np.linalg.norm(v10 - r * v12)

    def calculateCWDistance(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        min_dist = 100
        for i in range(1, len(points)):
            dist = self.getDistLS(points[i - 1], points[i])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def distanceCriterion(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        dist = self.calculateCWDistance(points)
        score = (self.DIST_PARAM - dist) / self.DIST_PARAM
        return (0, score)[score > 0]

    def sectorCriterion(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        sector_angle = np.pi / self.N_SECTORS
        foi_sectors = self.FOI_ANGLE / sector_angle
        foi_range = ((self.N_SECTORS - foi_sectors) // 2, (self.N_SECTORS + foi_sectors) // 2 - 1)
        min_sector, max_sector = self.getMinMaxSectors(points, sector_angle)

        a = min_sector - foi_range[0]
        b = min_sector - foi_range[1]
        c = max_sector - foi_range[0]
        d = max_sector - foi_range[1]
        intersection = 0

        if a < 0 and c >= 0 and d <= 0:
            intersection = (c + 1) / (c - a + 1)
        if a >= 0 and d <= 0:
            intersection = 1
        if a >= 0 and b <= 0 and d > 0:
            intersection = (-b + 1) / (c - a + 1)
        if a < 0 and d > 0:
            intersection = (a - b) / (c - a + 1)

        return intersection

    def angleCriterion(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        min_angle, max_angle = self.getMinMaxAngles(points)
        foi_range = ((np.pi - self.FOI_ANGLE) / 2, (np.pi + self.FOI_ANGLE) / 2)

        a = min_angle - foi_range[0]
        b = min_angle - foi_range[1]
        c = max_angle - foi_range[0]
        d = max_angle - foi_range[1]
        intersection = 0

        if a < 0 and c >= 0 and d <= 0:
            intersection = (c + 1) / (c - a + 1)
        if a >= 0 and d <= 0:
            intersection = 1
        if a >= 0 and b <= 0 and d > 0:
            intersection = (-b + 1) / (c - a + 1)
        if a < 0 and d > 0:
            intersection = (a - b) / (c - a + 1)

        return intersection

    def calculateCWArea(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        area = 0
        for i in range(len(points)):
            area += (
                (points[i]["x"] - points[i - 1]["x"])
                * (points[i]["y"] + points[i - 1]["y"])
                / 2
            )
        return abs(area)

    def areaCriterion(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        area = calculateCWArea(points)
        score = area / self.AREA_PARAM
        return (1, score)[score < self.AREA_PARAM]

    def isValidCW(self, points):
        """Summary
        
        Args:
            points (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if len(points) > self.VERTEX_TH and self.calculateCWArea(points) > self.AREA_TH:
            return 1
        return 0

    def predict_turn(self, gyro_z_val):
        """Summary
        
        Args:
            gyro_z_val (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        turn = 0
        if gyro_z_val > self.YAW_TH:
            turn = 1
        elif gyro_z_val < -self.YAW_TH:
            turn = -1

        return turn

    def turnCriterion(self, points, turn):
        """Summary
        
        Args:
            points (TYPE): Description
            turn (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        y_centroid = sum(point["y"] for point in points) / len(points)
        if turn * y_centroid / self.CENTROID_TH > 1:
            return 1
        return 0

    def get_CW_scores_IMU(
        self,
        cw_list,
        ego_speed,
        turn,
    ):
        """Summary
        
        Args:
            cw_list (TYPE): Description
            ego_speed (TYPE): Description
            turn (TYPE): Description
        """
        cw_scores = []

        for i, cw in enumerate(cw_list):
            points = cw["polygon_bev"]["vertices"]

            if not (isValidCW(points, vertex_t)):
                cw_scores.append((i, 0))
                continue

            dist_score = distanceCriterion(points)
            area_score = areaCriterion(points)
            angle_score = angleCriterion(points)
            turn_score = 0
            if ego_speed < self.INTERSECTION_SPEED_TH:
                turn_score = self.turnCriterion(points, turn)

            score = (
                self.WTS[0] * dist_score
                + self.WTS[1] * angle_score
                + self.WTS[2] * area_score
                + self.WTS[3] * turn_score
            ) / sum(self.WTS)
            cw_scores.append((i, score))

        cw_scores.sort(key=lambda x: x[1], reverse=1)

        return cw_scores

    def polygon_area(self, xs, ys):
        """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        
        Args:
            xs (TYPE): Description
            ys (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # https://stackoverflow.com/a/30408825/7128154
        return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


	def risk_val_asses( tracking_id, ped_bev_x, ped_bev_y, rel_vel_x, rel_vel_y, dist_th_1, IS_IN_LANE):
    Ped_RA = []
    if IS_IN_LANE is True:
        dist = math.sqrt(ped_bev_x*ped_bev_x + ped_bev_y*ped_bev_y)
        if dist<dist_th_1:
            Ped_RA.append((1, tracking_id, (ped_bev_x, ped_bev_y)))
        else:
            dist_score = ((dist-dist_th_1)/dist_th_1)
            time = dist/math.sqrt(rel_vel_x*rel_vel_x + rel_vel_y*rel_vel_y)
            score = (dist_score + time)/2
            Risk_value = 1-score
            Ped_RA.append((Risk_value, tracking_id, (ped_bev_x, ped_bev_y)))
        return Ped_RA
    else:
        return Ped_RA

	def IS_IN_LANE(left_lane, right_lane, ped_pos):
    vertex = (left_lane).append(right_lane)
	ans = CVW(vertex)
    if ans == 2:
        return True
    else:
        return False

    def polygon_centroid(self, crosswalk):
        """https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
        
        Args:
            crosswalk (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        xs = []
        ys = []
        crosswalk_points_array
        for item in crosswalk["polygon_bev"]["verices"]:
            xs.append(item["x"])
            ys.append[item["y"]]
            crosswalk_points_array.append([x, y])
        xs = np.array(xs)
        ys = np.array(ys)
        xy = np.array([xs, ys])
        c = np.dot(
            xy + np.roll(xy, 1, axis=1), xs * np.roll(ys, 1) - np.roll(xs, 1) * ys
        ) / (6 * polygon_area(xs, ys))
        return c, np.array(crosswalk_points_array)

    def Unity_dir(self, item):
        """Summary
        
        Args:
            item (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        array = np.array(item["x"], item["y"])
        return array

    def check_inclusion(self, target_cw, ped_pos):
        """Summary
        
        Args:
            target_cw (TYPE): Description
            ped_pos (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        result = 0
        for i in range(len(target_cw)):
            l1 = target_cw[i] - ped_pos
            l2 = target_cw[(i + 1) % len(target_cw)] - ped_pos
            angle = math.acos(np.dot(l1, l2))
            reuslt += angle
        return abs(result) > 355

    def approach_diagnosis(self, arr):
        """Summary
        
        Args:
            arr (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        count = 0
        for i in range(len(arr) - 1):
            diff = arr[i] - arr[i + 1]
            arr[i] = diff
            if diff > self.min_diff:
                count += 1
            else:
                count -= 1
        return count

	def IS_IN_LANE(left_lane, right_lane, ped_pos):
    	vertex = (left_lane).append(right_lane)
		ans = CVW(vertex)
    	if ans == 2:
        	return True
    	else:
        	return False


    def accum_result(self, obj):
        """Summary
        
        Args:
            obj (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        ped_id_dict = {}
        m_results = []
        accumulated_result = {
            "ped_id": None,
            "m_theta1": 0,
            "m_theta2": 0,
            "m_theta3": 0,
            "m_theta4": 0,
            "m_distance_g": 0,
            "m_distance_v": 0,
            "m_inclusion": 0,
            "approach_cw": None,
        }
        for i, item in enumerate(obj):
            for j, x in enumerate(item):
                if x["ped_id"] in ped_id_dict.values():
                    ped_id_dict[str(x["ped_id"])].append((i, j))
                else:
                    ped_id_dict[str(x["ped_id"])] = []
                    ped_id_dict[str(x["ped_id"])].append((i, j))

        for key, value in ped_id_dict.items():
            n = len(value)
            n = 0
            m_theta1 = 0
            m_theta2 = 0
            m_theta3 = 0
            m_theta4 = 0
            m_distance_g = 0
            m_distance_v = 0
            m_inclusion = 0
            approach_cw = []
            accumulated_result["ped_id"] = key
            for x in value:
                i, j = x
                dict_ = obj[i][j]
                m_theta1 += dict_["theta1"]
                m_theta2 += dict_["theta2"]
                m_theta3 += dict_["theta3"]
                m_theta4 += dict_["theta4"]
                m_distance_g += dict_["distance_g"]
                m_distance_v += dict_["distance_v"]
                m_inclusion += dict_["inclusion"]
                approach_cw.append(dict_["distance_g"])
            accumulated_result["m_theta1"] = m_theta1 / n
            accumulated_result["m_theta2"] = m_theta2 / n
            accumulated_result["m_theta3"] = m_theta3 / n
            accumulated_result["m_theta4"] = m_theta4 / n
            accumulated_result["m_distance_g"] = m_distance_g / n
            accumulated_result["m_distance_v"] = m_distance_v / n
            accumulated_result["m_inclusion"] = m_inclusion / n
            accumulated_result["approach_cw"] = approach_cw
            m_results.append(accumulated_result)

        return m_results

    def pedestrian_analysis(self, crosswalks, objects, cw_scores):
        """Summary
        
        Args:
            crosswalks (TYPE): Description
            objects (TYPE): Description
            cw_scores (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        Ped_analysis_result = []
        self.pedestrianAnalysis_object.pop(0)
        for item in objects:
            count = 0
            count_face = 0
            count_body = 0
            theta1 = 0
            theta2 = 0
            theta3 = 0
            theta4 = 0
            m_distance_g = 0
            m_distance_v = 0
            inclusion = 0

            analysis_data = {
                "ped_id": None,
                "theta1": theta1,
                "theta2": theta2,
                "theta3": theta3,
                "theta4": theta4,
                "distance_g": m_distance_g,
                "distance_v": m_distance_v,
                "inclusion": inclusion,
                "approach_cw": 0,
                "PEDESTRIAN_BODY_DIRECTION": -1,
                "PEDESTRIAN_FACE_DIRECTION": -1,
                "IS_CROSSWALK_PROXIMITY": -1,
                "IS_APPROACHING_CROSSWALK": -1,
                "CROSSING_INTENTION": -1,
            }
            if item["recognized_class"] == "PEDESTRIAN" and len(cw_scores) != 0:
                attr = item["pedestrian_attr"]
                analysis_data["ped_id"] = item["object_id"]

                camera_direction_vector = np.array(
                    [-item["location_bev"]["x"], -item["location_bev"]["y"]]
                )
                camera_direction_vector = (
                    camera_direction_vector
                    / (camera_direction_vector ** 2).sum() ** 0.5
                )
                pedestrian_position = np.array(
                    [item["location_bev"]["x"], item["location_bev"]["y"]]
                )

                crosswalk_position_g, target_crosswalk = polygon_centroid(
                    crosswalks[cw_scores[0][0]]
                )
                crosswalk_direction_vector = crosswalk_position_g - pedestrian_position
                crosswalk_direction_vector = (
                    crosswalk_direction_vector
                    / (crosswalk_direction_vector ** 2).sum() ** 0.5
                )
                crosswalk_direction_angle = math.degrees(
                    math.atan2(
                        crosswalk_direction_vector[1], crosswalk_direction_vector[0]
                    )
                )
                distance_g = np.linalg.norm(pedestrian_position - crosswalk_position_g)
                distance_v = math.sqrt(
                    min(
                        map(
                            lambda b: (b[0] - pedestrian_position[0]) ** 2
                            + (b[1] - pedestrian_position[1]) ** 2,
                            target_crosswalk,
                        )
                    )
                )
                inclusion = check_inclusion(target_crosswalk, pedestrian_position)

                if not (
                    attr["face_direction_bev"]["x"] == 0
                    and attr["face_direction_bev"]["y"] == 0
                ):
                    count_face += 1
                    face_direction_unity = self.Unity_dir(attr["face_direction_bev"])
                    theta1 += math.degrees(
                        math.acos(np.dot(camera_direction_vector, face_direction_unity))
                    )
                    theta2 += math.degrees(
                        math.acos(
                            np.dot(crosswalk_direction_vector, face_direction_unity)
                        )
                    )

                if not (
                    attr["body_direction_bev"]["x"] == 0
                    and attr["body_direction_bev"]["y"] == 0
                ):
                    count_body += 1
                    body_direction_unity = self.Unity_dir(attr["body_direction_bev"])
                    theta3 += math.degrees(
                        math.acos(np.dot(camera_direction_vector, body_direction_unity))
                    )
                    theta4 += math.degrees(
                        math.acos(
                            np.dot(crosswalk_direction_vector, body_direction_unity)
                        )
                    )

                analysis_data["distance_g"] = distance_g
                analysis_data["distance_v"] = distance_v
                analysis_data["theta1"] = theta1
                analysis_data["theta2"] = theta2
                analysis_data["theta3"] = theta3
                analysis_data["theta4"] = theta4
                analysis_data["approach_cw"] = distance_g
                analysis_data["inclusion"] = inclusion
            Ped_analysis_result.append(analysis_data)
        self.pedestrianAnalysis_object.append(Ped_analysis_result)

        accum_analysis = accum_result(self.pedestrianAnalysis_object)
        RESULTS = [{} for i in range(len(accum_analysis))]

        for idx, item in enumerate(accum_analysis):
            pedestrian_id = item["ped_id"]
            RESULTS[idx]["ped_id"] = pedestrian_id
            m_theta1 = item["m_theta1"]
            m_theta2 = item["m_theta2"]
            m_theta3 = item["m_theta3"]
            m_theta4 = item["m_theta4"]
            m_inclusion = item["m_inclusion"]
            approach_cw = item["approach_cw"]
            m_distance_v = item["m_distance_v"]

            if abs(m_theta1) < self.Faceth1:
                PEDESTRIAN_FACE_DIRECTION = 1

            elif abs(m_theta2) < self.Faceth2:
                PEDESTRIAN_FACE_DIRECTION = 2
            else:
                PEDESTRIAN_FACE_DIRECTION = 0

            analysis_data["ped_id"][
                "PEDESTRIAN_FACE_DIRECTION"
            ] = PEDESTRIAN_FACE_DIRECTION
            RESULTS[idx]["PEDESTRIAN_FACE_DIRECTION"] = PEDESTRIAN_FACE_DIRECTION

            if abs(m_theta3) < self.Bodyth1:
                PEDESTRIAN_BODY_DIRECTION = 1

            elif abs(m_theta4) < self.Bodyth2:
                PEDESTRIAN_BODY_DIRECTION = 2
            else:
                PEDESTRIAN_BODY_DIRECTION = 0
            analysis_data["ped_id"][
                "PEDESTRIAN_BODY_DIRECTION"
            ] = PEDESTRIAN_BODY_DIRECTION
            RESULTS[idx]["PEDESTRIAN_BODY_DIRECTION"] = PEDESTRIAN_BODY_DIRECTION

            # Proximity Check
            if m_inclusion > self.inclusion_th:
                IS_CROSSWALK_PROXIMITY = 2
            elif m_distance_v < self.crosswalk_th:
                IS_CROSSWALK_PROXIMITY = 1
            else:
                IS_CROSSWALK_PROXIMITY = 0

            analysis_data["ped_id"]["IS_CROSSWALK_PROXIMITY"] = IS_CROSSWALK_PROXIMITY
            RESULTS[idx]["IS_CROSSWALK_PROXIMITY"] = IS_CROSSWALK_PROXIMITY

            # approach diagnosis
            if not len(self.pedestrianAnalysis_object) < self.number_of_frames:
                approach_diagnosis_result = approach_diagnosis(approach_cw)
                if approach_diagnosis_result > self.Approaching_Th1:
                    IS_APPROACHING_CROSSWALK = 0
                else:
                    IS_APPROACHING_CROSSWALK = 1
            else:
                IS_APPROACHING_CROSSWALK = 0
            RESULTS[idx]["IS_APPROACHING_CROSSWALK"] = IS_APPROACHING_CROSSWALK

        return RESULTS
