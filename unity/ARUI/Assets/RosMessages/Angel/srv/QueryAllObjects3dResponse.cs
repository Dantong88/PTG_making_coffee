//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Angel
{
    [Serializable]
    public class QueryAllObjects3dResponse : Message
    {
        public const string k_RosMessageName = "angel_msgs/QueryAllObjects3d";
        public override string RosMessageName => k_RosMessageName;

        public AruiObject3dMsg[] all_objects;

        public QueryAllObjects3dResponse()
        {
            this.all_objects = new AruiObject3dMsg[0];
        }

        public QueryAllObjects3dResponse(AruiObject3dMsg[] all_objects)
        {
            this.all_objects = all_objects;
        }

        public static QueryAllObjects3dResponse Deserialize(MessageDeserializer deserializer) => new QueryAllObjects3dResponse(deserializer);

        private QueryAllObjects3dResponse(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.all_objects, AruiObject3dMsg.Deserialize, deserializer.ReadLength());
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.WriteLength(this.all_objects);
            serializer.Write(this.all_objects);
        }

        public override string ToString()
        {
            return "QueryAllObjects3dResponse: " +
            "\nall_objects: " + System.String.Join(", ", all_objects.ToList());
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register()
        {
            MessageRegistry.Register(k_RosMessageName, Deserialize, MessageSubtopic.Response);
        }
    }
}