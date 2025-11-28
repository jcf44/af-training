"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, RefreshCw, Upload } from "lucide-react";

import { API_URL } from "@/utils/config";

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function DatasetsPage() {
    const { data, error, isLoading, mutate } = useSWR(`${API_URL}/datasets/`, fetcher);
    const [isPrepareOpen, setIsPrepareOpen] = useState(false);
    const [isUploadOpen, setIsUploadOpen] = useState(false);
    const [datasetName, setDatasetName] = useState("");
    const [uploadFile, setUploadFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);

    const handlePrepare = async () => {
        // TODO: Implement prepare API call
        console.log("Preparing dataset:", datasetName);
        setIsPrepareOpen(false);
    };

    const handleUpload = async () => {
        if (!uploadFile || !datasetName) {
            alert("Please provide a name and select a zip file");
            return;
        }
        setIsUploading(true);
        const formData = new FormData();
        formData.append("file", uploadFile);
        formData.append("name", datasetName);

        try {
            const res = await fetch(`${API_URL}/datasets/upload`, {
                method: "POST",
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Upload failed");
            }
            mutate();
            setIsUploadOpen(false);
            setDatasetName("");
            setUploadFile(null);
            alert("Dataset uploaded successfully!");
        } catch (e: any) {
            console.error(e);
            alert(`Upload failed: ${e.message}`);
        } finally {
            setIsUploading(false);
        }
    };

    if (error) return <div>Failed to load datasets</div>;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Datasets</h1>
                    <p className="text-muted-foreground">Manage your training datasets.</p>
                </div>
                <div className="flex gap-2">
                    <Button variant="outline" size="icon" onClick={() => mutate()}>
                        <RefreshCw className="h-4 w-4" />
                    </Button>

                    <Dialog open={isUploadOpen} onOpenChange={setIsUploadOpen}>
                        <DialogTrigger asChild>
                            <Button variant="secondary">
                                <Upload className="mr-2 h-4 w-4" />
                                Upload Zip
                            </Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Upload Dataset</DialogTitle>
                                <DialogDescription>
                                    Upload a zip file containing YOLO format dataset (images/labels and data.yaml).
                                </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="upload-name" className="text-right">Name</Label>
                                    <Input
                                        id="upload-name"
                                        value={datasetName}
                                        onChange={(e) => setDatasetName(e.target.value)}
                                        className="col-span-3"
                                        placeholder="e.g., my-dataset"
                                    />
                                </div>
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="file" className="text-right">Zip File</Label>
                                    <Input
                                        id="file"
                                        type="file"
                                        accept=".zip"
                                        onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                                        className="col-span-3"
                                    />
                                </div>
                            </div>
                            <DialogFooter>
                                <Button onClick={handleUpload} disabled={isUploading}>
                                    {isUploading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                                    Upload
                                </Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>

                    <Dialog open={isPrepareOpen} onOpenChange={setIsPrepareOpen}>
                        <DialogTrigger asChild>
                            <Button>Prepare Raw</Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Prepare Dataset</DialogTitle>
                                <DialogDescription>
                                    Split raw images into train/val/test and generate YAML config.
                                </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                                <div className="grid grid-cols-4 items-center gap-4">
                                    <Label htmlFor="name" className="text-right">
                                        Name
                                    </Label>
                                    <Input
                                        id="name"
                                        value={datasetName}
                                        onChange={(e) => setDatasetName(e.target.value)}
                                        className="col-span-3"
                                        placeholder="e.g., ppe"
                                    />
                                </div>
                            </div>
                            <DialogFooter>
                                <Button onClick={handlePrepare}>Start Preparation</Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>
                </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <Card>
                    <CardHeader>
                        <CardTitle>Raw Datasets</CardTitle>
                        <CardDescription>Datasets in `datasets/raw`</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {isLoading ? (
                            <Loader2 className="h-6 w-6 animate-spin" />
                        ) : (
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Name</TableHead>
                                        <TableHead className="text-right">Action</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {data?.raw.map((name: string) => (
                                        <TableRow key={name}>
                                            <TableCell className="font-medium">{name}</TableCell>
                                            <TableCell className="text-right">
                                                <Badge variant="secondary">Raw</Badge>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                    {data?.raw.length === 0 && (
                                        <TableRow>
                                            <TableCell colSpan={2} className="text-center text-muted-foreground">No raw datasets found</TableCell>
                                        </TableRow>
                                    )}
                                </TableBody>
                            </Table>
                        )}
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader>
                        <CardTitle>Processed Datasets</CardTitle>
                        <CardDescription>Ready for training in `datasets/processed`</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {isLoading ? (
                            <Loader2 className="h-6 w-6 animate-spin" />
                        ) : (
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Name</TableHead>
                                        <TableHead className="text-right">Status</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {data?.processed.map((name: string) => (
                                        <TableRow key={name}>
                                            <TableCell className="font-medium">{name}</TableCell>
                                            <TableCell className="text-right">
                                                <Badge variant="default">Ready</Badge>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                    {data?.processed.length === 0 && (
                                        <TableRow>
                                            <TableCell colSpan={2} className="text-center text-muted-foreground">No processed datasets found</TableCell>
                                        </TableRow>
                                    )}
                                </TableBody>
                            </Table>
                        )}
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Dataset Configs</CardTitle>
                    <CardDescription>YAML configurations in `training/configs/datasets`</CardDescription>
                </CardHeader>
                <CardContent>
                    {isLoading ? (
                        <Loader2 className="h-6 w-6 animate-spin" />
                    ) : (
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Filename</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {data?.configs.map((name: string) => (
                                    <TableRow key={name}>
                                        <TableCell className="font-medium">{name}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
