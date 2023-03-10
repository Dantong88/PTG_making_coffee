//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Angel
{
    [Serializable]
    public class QueryTaskGraphResponse : Message
    {
        public const string k_RosMessageName = "angel_msgs/QueryTaskGraph";
        public override string RosMessageName => k_RosMessageName;

        public string task_title;
        public TaskGraphMsg task_graph;

        public QueryTaskGraphResponse()
        {
            this.task_title = "";
            this.task_graph = new TaskGraphMsg();
        }

        public QueryTaskGraphResponse(string task_title, TaskGraphMsg task_graph)
        {
            this.task_title = task_title;
            this.task_graph = task_graph;
        }

        public static QueryTaskGraphResponse Deserialize(MessageDeserializer deserializer) => new QueryTaskGraphResponse(deserializer);

        private QueryTaskGraphResponse(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.task_title);
            this.task_graph = TaskGraphMsg.Deserialize(deserializer);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.task_title);
            serializer.Write(this.task_graph);
        }

        public override string ToString()
        {
            return "QueryTaskGraphResponse: " +
            "\ntask_title: " + task_title.ToString() +
            "\ntask_graph: " + task_graph.ToString();
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
