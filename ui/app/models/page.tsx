"use client";

import useSWR from "swr";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, RefreshCw, FileBox, Loader2, Package } from "lucide-react";
import { toast } from "sonner";

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { useState } from "react";
import { useJobContext } from "@/contexts/job-context";

import { API_URL } from "@/utils/config";

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function ModelsPage() {
    const { data, error, mutate } = useSWR(`${API_URL}/models/`, fetcher);
    const { data: datasets } = useSWR(`${API_URL}/datasets/`, fetcher);
    const { addJob, isJobActive } = useJobContext();

    const [calibOpen, setCalibOpen] = useState(false);
    const [selectedModel, setSelectedModel] = useState<string | null>(null);
    const [selectedConfig, setSelectedConfig] = useState<string>("auto");

    const handleDownload = (filename: string, type: string) => {
        window.open(`${API_URL}/models/${filename}/download?type=${type}`, "_blank");
    };

    const handleExport = async (name: string) => {
        try {
            const res = await fetch(`${API_URL}/models/${name}/export`, { method: "POST" });
            if (!res.ok) throw new Error("Export failed");
            addJob(`export-${name}`);
            toast.info(`Export started for ${name}. You will be notified when it completes.`);
            mutate();
        } catch (e) {
            console.error(e);
            toast.error("Failed to start export");
        }
    };

    const handleCalibrate = async () => {
        if (!selectedModel || !selectedConfig) return;
        try {
            const res = await fetch(`${API_URL}/models/${selectedModel}/calibrate?config=${selectedConfig}`, { method: "POST" });
            if (!res.ok) throw new Error("Calibration failed");
            addJob(`calibrate-${selectedModel}`);
            toast.info(`Calibration started for ${selectedModel}. You will be notified when it completes.`);
            setCalibOpen(false);
        } catch (e) {
            console.error(e);
            toast.error("Failed to start calibration");
        }
    };

    if (error) return <div>Failed to load models</div>;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Models</h1>
                    <p className="text-muted-foreground">Download trained models and ONNX exports.</p>
                </div>
                <Button variant="outline" size="icon" onClick={() => mutate()}>
                    <RefreshCw className="h-4 w-4" />
                </Button>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <Card>
                    <CardHeader>
                        <CardTitle>Trained Models (PyTorch)</CardTitle>
                        <CardDescription>Best weights from training runs</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Name</TableHead>
                                    <TableHead className="text-right">Action</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {data?.trained.map((model: any) => (
                                    <TableRow key={model.name}>
                                        <TableCell className="font-medium flex items-center gap-2">
                                            <FileBox className="h-4 w-4 text-blue-500" />
                                            {model.name}
                                        </TableCell>
                                        <TableCell className="text-right">
                                            <div className="flex justify-end gap-2">
                                                <Button size="sm" variant="outline" onClick={() => handleDownload(model.name, "pt")}>
                                                    <Download className="mr-2 h-4 w-4" />
                                                    .pt
                                                </Button>
                                                <Button
                                                    size="sm"
                                                    variant="secondary"
                                                    onClick={() => handleExport(model.name)}
                                                    disabled={isJobActive(`export-${model.name}`)}
                                                >
                                                    {isJobActive(`export-${model.name}`) ? (
                                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                    ) : (
                                                        <RefreshCw className="mr-2 h-4 w-4" />
                                                    )}
                                                    Export ONNX
                                                </Button>
                                                <Button
                                                    size="sm"
                                                    variant="default"
                                                    onClick={() => { setSelectedModel(model.name); setCalibOpen(true); }}
                                                    disabled={isJobActive(`calibrate-${model.name}`)}
                                                >
                                                    {isJobActive(`calibrate-${model.name}`) ? (
                                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                    ) : (
                                                        <RefreshCw className="mr-2 h-4 w-4" />
                                                    )}
                                                    Calibrate INT8
                                                </Button>
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                ))}
                                {data?.trained.length === 0 && (
                                    <TableRow>
                                        <TableCell colSpan={2} className="text-center text-muted-foreground">No trained models found</TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Exported Models (ONNX)</CardTitle>
                        <CardDescription>Optimized models for deployment</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Filename</TableHead>
                                    <TableHead className="text-right">Action</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {data?.onnx.map((model: any) => (
                                    <TableRow key={model.name}>
                                        <TableCell className="font-medium flex items-center gap-2">
                                            <FileBox className="h-4 w-4 text-green-500" />
                                            {model.name}
                                        </TableCell>
                                        <TableCell className="text-right flex justify-end gap-2">
                                            <Button size="sm" variant="outline" onClick={() => handleDownload(model.name, "onnx")}>
                                                <Download className="mr-2 h-4 w-4" />
                                                .onnx
                                            </Button>
                                            <Button size="sm" variant="default" onClick={() => window.open(`${API_URL}/models/${model.name.replace('_best.onnx', '').replace('.onnx', '')}/bundle`, "_blank")}>
                                                <Package className="mr-2 h-4 w-4" />
                                                Bundle
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                                {data?.onnx.length === 0 && (
                                    <TableRow>
                                        <TableCell colSpan={2} className="text-center text-muted-foreground">No ONNX models found</TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>

                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Calibration Caches (INT8)</CardTitle>
                        <CardDescription>Cache files for INT8 quantization on DeepStream</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Filename</TableHead>
                                    <TableHead className="text-right">Action</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {data?.calibration?.map((file: any) => (
                                    <TableRow key={file.name}>
                                        <TableCell className="font-medium flex items-center gap-2">
                                            <FileBox className="h-4 w-4 text-purple-500" />
                                            {file.name}
                                        </TableCell>
                                        <TableCell className="text-right">
                                            <Button size="sm" variant="outline" onClick={() => handleDownload(file.name, "calibration")}>
                                                <Download className="mr-2 h-4 w-4" />
                                                Download .cache
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                                {(!data?.calibration || data?.calibration.length === 0) && (
                                    <TableRow>
                                        <TableCell colSpan={2} className="text-center text-muted-foreground">No calibration files found</TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </CardContent>
                </Card>
            </div>

            <Dialog open={calibOpen} onOpenChange={setCalibOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Calibrate Model: {selectedModel}</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                        <div className="space-y-2">
                            <Label>Calibration Dataset</Label>
                            <Select value={selectedConfig} onValueChange={setSelectedConfig}>
                                <SelectTrigger>
                                    <SelectValue placeholder="Select config" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="auto">✨ Auto-generate from Training Data</SelectItem>
                                    {datasets?.calibration?.map((config: string) => (
                                        <SelectItem key={config} value={config}>{config}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                            <p className="text-sm text-muted-foreground">
                                Select "Auto-generate" to automatically use the validation set from the training data.
                            </p>
                        </div>
                        <Button className="w-full" onClick={handleCalibrate} disabled={!selectedConfig}>
                            Start Calibration
                        </Button>
                    </div>
                </DialogContent>
            </Dialog>
        </div >
    );
}
