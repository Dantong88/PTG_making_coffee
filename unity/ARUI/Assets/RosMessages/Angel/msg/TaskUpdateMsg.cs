//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Angel
{
    [Serializable]
    public class TaskUpdateMsg : Message
    {
        public const string k_RosMessageName = "angel_msgs/TaskUpdate";
        public override string RosMessageName => k_RosMessageName;

        // 
        //  Represents the current status of the task being performed.
        // 
        //  Standard ROS message header
        public Std.HeaderMsg header;
        //  Task name
        public string task_name;
        //  Task description
        public string task_description;
        //  Items required
        public TaskItemMsg[] task_items;
        //  List of steps for this task
        //  DEPRECATED: Users should instead query the QueryTaskGraph service.
        public string[] steps;
        //  The index of the step currently in progress.
        //  A value of `-1` indicates that no step has been started yet.
        public sbyte current_step_id;
        //  String of the step currently in progress.
        public string current_step;
        //  Previous step is the step worked on before the current step.
        public string previous_step;
        //  Current activity classification prediction
        public string current_activity;
        public string next_activity;
        //  Time remaining to move to next task (e.g. waiting for tea to steep)
        //  -1 means that this is not a time based task
        public int time_remaining_until_next_task;
        //  Current confidence from the HMM that a recipe is complete.
        //  Only task_monitor_v2 will fill this in.
        public float task_complete_confidence;
        //  Array of length n_steps (same length as `steps` above) that indicates which
        //  steps, by index association, are complete.
        public bool[] completed_steps;

        public TaskUpdateMsg()
        {
            this.header = new Std.HeaderMsg();
            this.task_name = "";
            this.task_description = "";
            this.task_items = new TaskItemMsg[0];
            this.steps = new string[0];
            this.current_step_id = 0;
            this.current_step = "";
            this.previous_step = "";
            this.current_activity = "";
            this.next_activity = "";
            this.time_remaining_until_next_task = 0;
            this.task_complete_confidence = 0.0f;
            this.completed_steps = new bool[0];
        }

        public TaskUpdateMsg(Std.HeaderMsg header, string task_name, string task_description, TaskItemMsg[] task_items, string[] steps, sbyte current_step_id, string current_step, string previous_step, string current_activity, string next_activity, int time_remaining_until_next_task, float task_complete_confidence, bool[] completed_steps)
        {
            this.header = header;
            this.task_name = task_name;
            this.task_description = task_description;
            this.task_items = task_items;
            this.steps = steps;
            this.current_step_id = current_step_id;
            this.current_step = current_step;
            this.previous_step = previous_step;
            this.current_activity = current_activity;
            this.next_activity = next_activity;
            this.time_remaining_until_next_task = time_remaining_until_next_task;
            this.task_complete_confidence = task_complete_confidence;
            this.completed_steps = completed_steps;
        }

        public static TaskUpdateMsg Deserialize(MessageDeserializer deserializer) => new TaskUpdateMsg(deserializer);

        private TaskUpdateMsg(MessageDeserializer deserializer)
        {
            this.header = Std.HeaderMsg.Deserialize(deserializer);
            deserializer.Read(out this.task_name);
            deserializer.Read(out this.task_description);
            deserializer.Read(out this.task_items, TaskItemMsg.Deserialize, deserializer.ReadLength());
            deserializer.Read(out this.steps, deserializer.ReadLength());
            deserializer.Read(out this.current_step_id);
            deserializer.Read(out this.current_step);
            deserializer.Read(out this.previous_step);
            deserializer.Read(out this.current_activity);
            deserializer.Read(out this.next_activity);
            deserializer.Read(out this.time_remaining_until_next_task);
            deserializer.Read(out this.task_complete_confidence);
            deserializer.Read(out this.completed_steps, sizeof(bool), deserializer.ReadLength());
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.header);
            serializer.Write(this.task_name);
            serializer.Write(this.task_description);
            serializer.WriteLength(this.task_items);
            serializer.Write(this.task_items);
            serializer.WriteLength(this.steps);
            serializer.Write(this.steps);
            serializer.Write(this.current_step_id);
            serializer.Write(this.current_step);
            serializer.Write(this.previous_step);
            serializer.Write(this.current_activity);
            serializer.Write(this.next_activity);
            serializer.Write(this.time_remaining_until_next_task);
            serializer.Write(this.task_complete_confidence);
            serializer.WriteLength(this.completed_steps);
            serializer.Write(this.completed_steps);
        }

        public override string ToString()
        {
            return "TaskUpdateMsg: " +
            "\nheader: " + header.ToString() +
            "\ntask_name: " + task_name.ToString() +
            "\ntask_description: " + task_description.ToString() +
            "\ntask_items: " + System.String.Join(", ", task_items.ToList()) +
            "\nsteps: " + System.String.Join(", ", steps.ToList()) +
            "\ncurrent_step_id: " + current_step_id.ToString() +
            "\ncurrent_step: " + current_step.ToString() +
            "\nprevious_step: " + previous_step.ToString() +
            "\ncurrent_activity: " + current_activity.ToString() +
            "\nnext_activity: " + next_activity.ToString() +
            "\ntime_remaining_until_next_task: " + time_remaining_until_next_task.ToString() +
            "\ntask_complete_confidence: " + task_complete_confidence.ToString() +
            "\ncompleted_steps: " + System.String.Join(", ", completed_steps.ToList());
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register()
        {
            MessageRegistry.Register(k_RosMessageName, Deserialize);
        }
    }
}
