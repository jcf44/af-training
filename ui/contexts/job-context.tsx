"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

interface JobContextType {
    activeJobs: Set<string>;
    addJob: (jobId: string) => void;
    removeJob: (jobId: string) => void;
    isJobActive: (jobId: string) => boolean;
    logs: string[];
    addLog: (lines: string[]) => void;
}

const JobContext = createContext<JobContextType | undefined>(undefined);

export function JobProvider({ children }: { children: ReactNode }) {
    const [activeJobs, setActiveJobs] = useState<Set<string>>(new Set());
    const [logs, setLogs] = useState<string[]>([]);

    const addJob = (jobId: string) => {
        setActiveJobs((prev) => {
            const newSet = new Set(prev);
            newSet.add(jobId);
            return newSet;
        });
        // Clear logs when starting a new job (optional, or keep history)
        setLogs([]);
    };

    const removeJob = (jobId: string) => {
        setActiveJobs((prev) => {
            const newSet = new Set(prev);
            newSet.delete(jobId);
            return newSet;
        });
    };

    const addLog = (lines: string[]) => {
        setLogs((prev) => [...prev, ...lines]);
    };

    const isJobActive = (jobId: string) => activeJobs.has(jobId);

    return (
        <JobContext.Provider value={{ activeJobs, addJob, removeJob, isJobActive, logs, addLog }}>
            {children}
        </JobContext.Provider>
    );
}

export function useJobContext() {
    const context = useContext(JobContext);
    if (context === undefined) {
        throw new Error("useJobContext must be used within a JobProvider");
    }
    return context;
}
