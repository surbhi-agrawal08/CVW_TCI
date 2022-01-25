"""The file contains code for the Turn Crosswalk Pedestrian Indication.
"""
import math


class TCI(object):

    """The class to manage our DASHCAM warning for Turn Crisswalk Indication system.
    
    Attributes:
        LOC_X_BWD_THRESH (float): Description
        LOC_X_FWD_THRESH (float): Description
        LOC_Y_LEFT_THRESH (float): Description
        LOC_Y_RIGHT_THRESH (float): Description
        TCI_USER_ACTIVE (bool): Description
    """

    def __init__(self, tci_config):
        """Initialize the TCI class.
        
        Args:
            tci_config (dict): The set of parameters for TCI passed from the config file.
        """
        self.init_config(tci_config)

    def init_config(self, tci_config):
        """Initialize the set of internal TCI parameters
        
        Args:
            tci_config (dict): The set of parameters for TCI passed from the config file.
        """
        self.LOC_X_FWD_THRESH = tci_config["LOC_X_FWD_THRESH"]
        self.LOC_X_BWD_THRESH = tci_config["LOC_X_BWD_THRESH"]
        self.LOC_Y_LEFT_THRESH = tci_config["LOC_Y_LEFT_THRESH"]
        self.LOC_Y_RIGHT_THRESH = tci_config["LOC_Y_RIGHT_THRESH"]
        self.TCI_USER_ACTIVE = tci_config["TCI_USER_ACTIVE"]

    def check_tci_active(self) -> bool:
        """Check if we need to raise the TCI alert or not.
        
        Returns:
            bool: Returns True/False based on simple logic checks.
        """
        return self.TCI_USER_ACTIVE

    def distance_check(self, ped_loc_x, ped_loc_y):
        """Check if the location of the pedestrian is within our threshold bounds or not.
        
        Args:
            ped_loc_x (float): x-component of location of pedestrian
            ped_loc_y (float): y-component of location of pedestrian
        
        Returns:
            float: Retuns True if pedestrain is within threshold bounds else return false.
        """
        if not (self.LOC_X_FWD_THRESH < ped_loc_x < self.LOC_X_BWD_THRESH) and not (
            self.LOC_Y_LEFT_THRESH < ped_loc_y < self.LOC_Y_RIGHT_THRESH
        ):
            return False
        return True

    def get_tci_location(self, tracking_id, ped_loc_x, ped_loc_y):
        """Summary
        
        Args:
            tracking_id (int): Tracking ID of pedestrian we are focussing on.
            ped_loc_x (float): x-component of location of pedestrian
            ped_loc_y (float): y-component of location of pedestrian
        
        Returns:
            tuple: The tuple of id, (location), (coordinates) of the pedestrians
        """
        if self.distance_check(ped_loc_x, ped_loc_y):
            angle = 180 * (math.atan(ped_loc_y / ped_loc_x)) / math.pi
            if -10.0 <= angle < 10.0 and ped_loc_x > 0.0:
                return (tracking_id, (1, "East"), (ped_loc_x, ped_loc_y))
            elif (80.0 <= angle or angle < -80.0) and ped_loc_y > 0.0:
                return (tracking_id, (3, "North"), (ped_loc_x, ped_loc_y))
            elif -10.0 <= angle < 10.0 and ped_loc_x < 0.0:
                return (tracking_id, (5, "West"), (ped_loc_x, ped_loc_y))
            elif (80.0 <= angle or angle > -80.0) and ped_loc_y < 0.0:
                return (tracking_id, (6, "South"), (ped_loc_x, ped_loc_y))
            elif 10.0 <= angle < 80.0 and ped_loc_x > 0 and ped_loc_y > 0.0:
                return (tracking_id, (2, "North-East"), (ped_loc_x, ped_loc_y))
            elif -80.0 <= angle < -10 and ped_loc_x < 0.0 and ped_loc_y > 0.0:
                return (tracking_id, (4, "North-West"), (ped_loc_x, ped_loc_y))
            elif 10.0 <= angle < 80.0 and ped_loc_x < 0.0 and ped_loc_y < 0.0:
                return (tracking_id, (7, "South-West"), (ped_loc_x, ped_loc_y))
            elif -10.0 < angle < 10.0 and ped_loc_x > 0.0 and ped_loc_y < 0.0:
                return (tracking_id, (8, "South-East"), (ped_loc_x, ped_loc_y))
        return None

    def raise_alert(self, objects):
        """The main function which raises the alert for this indication.
        
        Args:
            objects (list): List of objects passed from each recogntion frame to the HMI Manager.
        
        Returns:
            list: List of inforamtion of all pedestrians for which we need to raise indications.
        """
        tci_output = []
        for obj in objects:
            if obj["recognized_class"] == "PEDESTRIAN":
                location = self.get_tci_location(
                    self,
                    tracking_id=obj["tracking_id"],
                    ped_loc_x=obj["location_bev"]["x"],
                    ped_loc_y=obj["location_bev"]["y"],
                )
                if location:
                    tci_output.append(location)
        return tci_output
