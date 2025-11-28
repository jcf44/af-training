"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Terminal } from "lucide-react";
import { LogViewer } from "@/components/log-viewer";
import { JobProvider } from "@/contexts/job-context";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Toaster } from "sonner";
import { SSEListener } from "@/components/sse-listener";

export function ClientLayout({ children }: { children: React.ReactNode }) {
    const [logsOpen, setLogsOpen] = useState(false);

    return (
        <JobProvider>
            <SidebarProvider>
                <AppSidebar />
                <main className="w-full p-4 relative">
                    {children}
                    <div className="fixed bottom-4 right-4 z-40">
                        <Button
                            variant="default"
                            size="icon"
                            className="rounded-full h-12 w-12 shadow-lg bg-black hover:bg-gray-800 text-green-400 border border-green-900"
                            onClick={() => setLogsOpen(!logsOpen)}
                        >
                            <Terminal className="h-6 w-6" />
                        </Button>
                    </div>
                    <LogViewer open={logsOpen} onClose={() => setLogsOpen(false)} />
                </main>
                <Toaster />
                <SSEListener />
            </SidebarProvider>
        </JobProvider>
    );
}
