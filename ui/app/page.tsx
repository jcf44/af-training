import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Database, Brain, Box, ArrowRight } from "lucide-react";

export default function Home() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
      <p className="text-muted-foreground">
        Welcome to AF-Training. Manage your datasets, train models, and export them for deployment.
      </p>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Manage</div>
            <p className="text-xs text-muted-foreground">
              Prepare and split datasets
            </p>
            <Button asChild className="mt-4 w-full" variant="outline">
              <Link href="/datasets">
                Go to Datasets <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Training</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Train</div>
            <p className="text-xs text-muted-foreground">
              Start and monitor training jobs
            </p>
            <Button asChild className="mt-4 w-full" variant="outline">
              <Link href="/training">
                Go to Training <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Models</CardTitle>
            <Box className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Export</div>
            <p className="text-xs text-muted-foreground">
              Download and deploy models
            </p>
            <Button asChild className="mt-4 w-full" variant="outline">
              <Link href="/models">
                Go to Models <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
