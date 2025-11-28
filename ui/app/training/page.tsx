"use client";

import { useState } from "react";
import useSWR from "swr";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogDescription, DialogClose } from "@/components/ui/dialog";
import { Loader2, Play, Square, RefreshCw } from "lucide-react";

import { API_URL } from "@/utils/config";

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function TrainingPage() {
    const { data: jobs, error, mutate } = useSWR(`${API_URL}/train/jobs`, fetcher, { refreshInterval: 5000 });
    const { data: datasets } = useSWR(`${API_URL}/datasets/`, fetcher);

    const [formData, setFormData] = useState<any>({
        name: "my_model",
        model_type: "YOLO",
    });

    const { data: schema } = useSWR(
        formData.model_type ? `${API_URL}/train/schema/${formData.model_type}` : null,
        fetcher
    );

    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleStartTraining = async () => {
        if (formData.model_type === "YOLO" && !formData.dataset_config) {
            alert("Please select a dataset config");
            return;
        }
        setIsSubmitting(true);
        try {
            const res = await fetch(`${API_URL}/train/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });
            if (!res.ok) throw new Error("Failed to start training");
            mutate();
        } catch (e) {
            console.error(e);
            alert("Failed to start training");
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleStopJob = async (jobId: number) => {
        try {
            await fetch(`${API_URL}/train/${jobId}/stop`, { method: "POST" });
            mutate();
        } catch (e) {
            console.error(e);
        }
    };

    const [logsOpen, setLogsOpen] = useState(false);
    const [currentJobId, setCurrentJobId] = useState<number | null>(null);
    const { data: logsData } = useSWR(
        logsOpen && currentJobId ? `${API_URL}/train/${currentJobId}/logs` : null,
        fetcher,
        { refreshInterval: 2000 }
    );

    const [isClearDialogOpen, setIsClearDialogOpen] = useState(false);

    const handleClearJobs = () => {
        setIsClearDialogOpen(true);
    };

    const confirmClearJobs = async () => {
        try {
            const res = await fetch(`${API_URL}/train/jobs`, {
                method: "DELETE",
            });
            if (!res.ok) throw new Error("Failed to clear jobs");
            mutate();
            setIsClearDialogOpen(false);
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Training</h1>
                    <p className="text-muted-foreground">Train YOLO models on your datasets.</p>
                </div>
                <Button variant="outline" size="icon" onClick={() => mutate()}>
                    <RefreshCw className="h-4 w-4" />
                </Button>
            </div>

            <div className="grid gap-6 md:grid-cols-3">
                <Card className="md:col-span-1">
                    <CardHeader>
                        <CardTitle>New Training Job</CardTitle>
                        <CardDescription>Configure training parameters</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="name">Model Name</Label>
                            <Input
                                id="name"
                                value={formData.name}
                                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            />
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="type">Model Type</Label>
                            <Select
                                value={formData.model_type}
                                onValueChange={(val) => {
                                    setFormData({ ...formData, model_type: val });
                                    // Reset dynamic fields when type changes
                                    // We keep name and model_type
                                }}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Select type" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="YOLO">YOLO (Object Detection)</SelectItem>
                                    <SelectItem value="TimeSeries">Time Series (Forecasting)</SelectItem>
                                    <SelectItem value="Anomaly">Anomaly Detection (Maintenance)</SelectItem>
                                    <SelectItem value="Tabular">Tabular (Classification/Regression)</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        {/* Dynamic Form Fields */}
                        {schema && Object.entries(schema.properties).map(([key, field]: [string, any]) => {
                            // Skip fields if needed, but we render all properties
                            return (
                                <div key={key} className="space-y-2">
                                    <Label htmlFor={key}>{field.title || key}</Label>

                                    {/* Render Select for Enums */}
                                    {field.enum ? (
                                        <Select
                                            value={formData[key] !== undefined ? String(formData[key]) : String(field.default || "")}
                                            onValueChange={(val) => setFormData((prev: any) => ({ ...prev, [key]: val }))}
                                        >
                                            <SelectTrigger>
                                                <SelectValue placeholder={`Select ${field.title}`} />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {field.enum.map((opt: string, idx: number) => (
                                                    <SelectItem key={opt} value={opt}>
                                                        {field.enumNames ? field.enumNames[idx] : opt}
                                                    </SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    ) : field.enum_source === "datasets" ? (
                                        <Select
                                            value={formData[key] || ""}
                                            onValueChange={(val) => setFormData((prev: any) => ({ ...prev, [key]: val }))}
                                        >
                                            <SelectTrigger>
                                                <SelectValue placeholder="Select dataset" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {datasets?.configs.map((config: string) => (
                                                    <SelectItem key={config} value={config}>{config}</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                    ) : field.enum_source === "datasets_raw" ? (
                                        <Select
                                            value={formData[key] || ""}
                                            onValueChange={(val) => setFormData((prev: any) => ({ ...prev, [key]: val }))}
                                        >
                                            <SelectTrigger>
                                                <SelectValue placeholder="Select CSV dataset" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {/* TODO: Fetch raw datasets. For now using same list or mock */}
                                                <SelectItem value="sensor_data.csv">sensor_data.csv</SelectItem>
                                                <SelectItem value="production_log.csv">production_log.csv</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    ) : field.type === "integer" || field.type === "number" ? (
                                        <Input
                                            id={key}
                                            type="number"
                                            value={formData[key] !== undefined ? formData[key] : field.default}
                                            onChange={(e) => setFormData((prev: any) => ({ ...prev, [key]: parseInt(e.target.value) || 0 }))}
                                        />
                                    ) : (
                                        <Input
                                            id={key}
                                            type="text"
                                            value={formData[key] !== undefined ? formData[key] : (field.default || "")}
                                            onChange={(e) => setFormData((prev: any) => ({ ...prev, [key]: e.target.value }))}
                                        />
                                    )}
                                    {field.description && <p className="text-xs text-muted-foreground">{field.description}</p>}
                                </div>
                            );
                        })}

                        <Button className="w-full" onClick={handleStartTraining} disabled={isSubmitting}>
                            {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                            Start Training
                        </Button>
                    </CardContent>
                </Card>

                <Card className="md:col-span-2">
                    <CardHeader className="flex flex-row items-center justify-between">
                        <div>
                            <CardTitle>Training Jobs</CardTitle>
                            <CardDescription>History and active jobs</CardDescription>
                        </div>
                        <div className="flex gap-2">
                            <Button variant="outline" size="sm" onClick={handleClearJobs}>
                                Clear Jobs
                            </Button>
                            <Button variant="outline" size="icon" onClick={() => mutate()}>
                                <RefreshCw className="h-4 w-4" />
                            </Button>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <ScrollArea className="h-[600px]">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>ID</TableHead>
                                        <TableHead>Name</TableHead>
                                        <TableHead>Status</TableHead>
                                        <TableHead>Start Time</TableHead>
                                        <TableHead className="text-right">Action</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {jobs?.map((job: any) => (
                                        <TableRow key={job.id}>
                                            <TableCell>{job.id}</TableCell>
                                            <TableCell className="font-medium">{job.name}</TableCell>
                                            <TableCell>
                                                <div className="flex flex-col gap-1">
                                                    <Badge variant={
                                                        job.status === "running" ? "default" :
                                                            job.status === "completed" ? "secondary" :
                                                                job.status === "failed" ? "destructive" : "outline"
                                                    }>
                                                        {job.status}
                                                    </Badge>
                                                    {job.status === "running" && (
                                                        <Progress value={33} className="h-2 w-full" />
                                                        // TODO: Get actual progress from logs or API
                                                    )}
                                                </div>
                                            </TableCell>
                                            <TableCell>{new Date(job.start_time).toLocaleString()}</TableCell>
                                            <TableCell className="text-right">
                                                <div className="flex justify-end gap-2">
                                                    <Button variant="outline" size="sm" onClick={() => { setCurrentJobId(job.id); setLogsOpen(true); }}>
                                                        Logs
                                                    </Button>
                                                    {job.status === "running" && (
                                                        <Button variant="ghost" size="icon" onClick={() => handleStopJob(job.id)}>
                                                            <Square className="h-4 w-4 text-red-500" />
                                                        </Button>
                                                    )}
                                                </div>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                    {jobs?.length === 0 && (
                                        <TableRow>
                                            <TableCell colSpan={5} className="text-center text-muted-foreground">No jobs found</TableCell>
                                        </TableRow>
                                    )}
                                </TableBody>
                            </Table>
                        </ScrollArea>
                    </CardContent>
                </Card>
            </div>

            <Dialog open={logsOpen} onOpenChange={setLogsOpen}>
                <DialogContent className="max-w-3xl h-[80vh] flex flex-col">
                    <DialogHeader>
                        <DialogTitle>Training Logs - Job {currentJobId}</DialogTitle>
                    </DialogHeader>
                    <ScrollArea className="flex-1 min-h-0 w-full rounded-md border p-4 bg-black text-white font-mono text-sm">
                        <pre>{logsData?.logs || "Loading logs..."}</pre>
                    </ScrollArea>
                </DialogContent>
            </Dialog>
            <Dialog open={isClearDialogOpen} onOpenChange={setIsClearDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Clear All Jobs?</DialogTitle>
                        <DialogDescription>
                            This action cannot be undone. This will permanently delete all job history and stop any currently running training jobs.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsClearDialogOpen(false)}>Cancel</Button>
                        <Button variant="destructive" onClick={confirmClearJobs}>Clear All</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
