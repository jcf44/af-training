"use client";

import { useEffect, useRef, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { X, Terminal } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useJobContext } from "@/contexts/job-context";

interface LogViewerProps {
    open: boolean;
    onClose: () => void;
}

export function LogViewer({ open, onClose }: LogViewerProps) {
    const { logs } = useJobContext();
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [logs]);

    if (!open) return null;

    return (
        <div className="fixed bottom-4 right-4 w-[600px] z-50 shadow-2xl">
            <Card className="bg-black text-green-400 border-gray-800">
                <CardHeader className="flex flex-row items-center justify-between py-2 border-b border-gray-800">
                    <CardTitle className="text-sm font-mono flex items-center gap-2">
                        <Terminal className="h-4 w-4" />
                        Live Logs
                    </CardTitle>
                    <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-400 hover:text-white" onClick={onClose}>
                        <X className="h-4 w-4" />
                    </Button>
                </CardHeader>
                <CardContent className="p-0">
                    <ScrollArea className="h-[300px] p-4 font-mono text-xs">
                        {logs.length === 0 ? (
                            <div className="text-gray-500 italic">Waiting for logs...</div>
                        ) : (
                            logs.map((line, i) => (
                                <div key={i} className="whitespace-pre-wrap break-all">
                                    {line}
                                </div>
                            ))
                        )}
                        <div ref={scrollRef} />
                    </ScrollArea>
                </CardContent>
            </Card>
        </div>
    );
}
