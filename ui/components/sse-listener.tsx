"use client";

import { useEffect } from "react";
import { toast } from "sonner";
import { useSWRConfig } from "swr";

import { useJobContext } from "@/contexts/job-context";

import { API_URL } from "@/utils/config";

export function SSEListener() {
    const { mutate } = useSWRConfig();
    const { removeJob, addLog } = useJobContext();

    useEffect(() => {
        const eventSource = new EventSource(`${API_URL}/events`);

        eventSource.onopen = () => {
            console.log("SSE Connected");
        };

        eventSource.onmessage = (event) => {
            try {
                const parsed = JSON.parse(event.data);
                const { type, data } = parsed;

                console.log("SSE Event:", type, data);

                if (type === "job_completed") {
                    toast.success(`Training job "${data.name}" completed!`);
                    mutate(`${API_URL}/train/jobs`); // Refresh jobs list
                } else if (type === "export_completed") {
                    toast.success(`Export for "${data.name}" completed!`);
                    removeJob(`export-${data.name}`);
                    mutate(`${API_URL}/models/`); // Refresh models list
                } else if (type === "calibration_completed") {
                    toast.success(`Calibration for "${data.name}" completed!`);
                    removeJob(`calibrate-${data.name}`);
                    mutate(`${API_URL}/models/`);
                } else if (type === "log_update") {
                    addLog(data.lines);
                }
            } catch (e) {
                console.error("Error parsing SSE event:", e);
            }
        };

        eventSource.onerror = (e) => {
            console.error("SSE Error:", e);
            eventSource.close();
        };

        return () => {
            eventSource.close();
        };
    }, [mutate]);

    return null;
}
