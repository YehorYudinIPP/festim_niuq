
# TODO make a class to compute qualities of interest using FESTIM results

class Diagnostics:

    def __init__(self, results):
        """
        Initialize Diagnostics with FESTIM results.
        
        :param results: Results from the FESTIM simulation.
        """
        self.results = results

    def compute_qoi(self, qoi_name):
        """
        Compute a quality of interest (QoI) from the results.
        
        :param qoi_name: Name of the QoI to compute.
        :return: Computed QoI value.
        """
        # Placeholder for actual QoI computation logic
        return self.results.get(qoi_name, None)
    
    #TODO function to compute total tritium inverntory inside simualted volume
    def compute_total_tritium_inventory(self):
        """
        Compute the total tritium inventory in the simulated volume.
        
        :return: Total tritium inventory.
        """
        # Assuming results has a 'tritium_inventory' key with the inventory data
        return self.results.get('tritium_inventory', 0.0)