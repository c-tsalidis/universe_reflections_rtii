using UnityEngine;

public static class Calculate {
    public static Vector3 Add(Vector3 v1, Vector3 v2) {
        return new Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
    }
    
    public static Vector3 Subtract(Vector3 v1, Vector3 v2) {
        return new Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
    }
    
    public static Vector3 Multiply(float scalar, Vector3 vector) {
        return scalar * vector;
    }
    
    public static Vector3 Divide(float scalar, Vector3 vector) {
        return vector / scalar;
    }

    public static Vector3 Limit(Vector3 vector, float maxMagnitude) {
        // Limit the magnitude of this vector to the value used for the max parameter.
        var x = maxMagnitude / Vector3.Magnitude(vector);
        return vector = Multiply(x, vector);
    } 
    
    public static float Map(float value, float oldMin, float oldMax, float newMin, float newMax){
        float oldRange = (oldMax - oldMin);
        float newRange = (newMax - newMin);
        float newValue = (((value - oldMin) * newRange) / oldRange) + newMin;
        return(newValue);
    }
}