from typing import Dict, Any, List

class PublicSafetyAlert:
    """
    Public Safety Alert Engine that translates ethical risk scores and predictions
    into actionable policy alerts and emergency response recommendations.
    """
    def __init__(self):
        pass

    def generate_alert(
        self, 
        country: str, 
        predicted_cases: float, 
        confidence: float, 
        risk_score: float
    ) -> Dict[str, Any]:
        """
        Generates public safety alerts, recommendations, and protocols.
        """
        # Determine safety threat level
        if risk_score < 25.0:
            threat_level = "LOW RISK"
            color = "#2ecc71" # Green
            protocols = [
                "Continue standard hygiene practices and regular health monitoring.",
                "Maintain active surveillance systems at ports of entry.",
                "Ensure local clinics have baseline PPE supplies."
            ]
        elif risk_score < 50.0:
            threat_level = "MODERATE RISK"
            color = "#f1c40f" # Yellow
            protocols = [
                "Encourage voluntary mask-wearing in crowded indoor spaces.",
                "Advise high-risk individuals to avoid non-essential travel.",
                "Initiate public health campaigns detailing rapid testing locations.",
                "Increase daily PCR testing capacities in municipal clinics."
            ]
        elif risk_score < 75.0:
            threat_level = "HIGH RISK"
            color = "#e67e22" # Orange
            protocols = [
                "Mandate mask-wearing in public transport and government buildings.",
                "Restrict indoor gatherings to 50% capacity.",
                "Recommend work-from-home options where feasible.",
                "Prepare local hospitals to activate surge capacity plans."
            ]
        else:
            threat_level = "CRITICAL"
            color = "#e74c3c" # Red
            protocols = [
                "Enforce mandatory public lockdowns or localized quarantine zones.",
                "Suspend all large-scale public events and close non-essential venues.",
                "Deploy emergency hospital beds and activate military medical logistics.",
                "Implement strict inter-state/regional travel restrictions."
            ]
            
        confidence_pct = int(confidence * 100)
        
        return {
            "country": country,
            "predicted_cases": int(predicted_cases),
            "confidence": f"{confidence_pct}%",
            "risk_score": risk_score,
            "threat_level": threat_level,
            "color": color,
            "protocols": protocols
        }
